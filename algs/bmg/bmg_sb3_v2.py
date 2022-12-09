import sys
sys.path.append('../..')

import copy
import warnings
from collections import deque
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchopt
from gym import spaces

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy#, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.distributions import kl_divergence

from algs.policies.extractor import MultiInputActorCriticPolicy


class BMG(OnPolicyAlgorithm):
    """
    Implementation of Bootstrapped Meta-learning (Flennerhag et al., ICLR 2022)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            # Base PPO algorithm
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.4,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
            _init_setup_model: bool = True,

            # Meta-learning related parameters
            K: int = 1,
            L: int = 1,
            meta_window_size: int = 10,
            meta_learning_rate: Union[float, Schedule] = 1e-5,
            create_eval_env: bool = False,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Meta-learning related variables
        self.meta_learning_rate = meta_learning_rate
        self.K = K
        self.L = L
        self.meta_window_size = meta_window_size
        self.avg_rollout_reward_buffer = deque([0] * 10, maxlen=meta_window_size)

        self.batch_size = batch_size
        self.n_epochs = K + L
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        # define meta_parameter and its optimizer
        self.vf_coef = nn.Parameter(torch.tensor(0.4, requires_grad=True))
        self.ent_coef = nn.Parameter(torch.tensor(0.00, requires_grad=True))
        #self.ent_coef = 0
        self.policy_coef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.coef_optimizer = torch.optim.Adam(
                    [self.policy_coef] + [self.vf_coef] + [self.ent_coef],
                    lr=self.meta_learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-4
        )

        self.pretrain_step = 2_000_000 // n_steps // 10 * (K + L)
        print(self.pretrain_step)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # For PPO implementation
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        # Meta-learning related variables
        self.kth_policy = copy.deepcopy(self.policy)
        self.meta_learner = MLP(input_size=self.meta_window_size, feature_size=32, device=self.policy.device)
        self.meta_optimizer = torch.optim.Adam(
            self.meta_learner.parameters(),
            lr=self.meta_learning_rate, betas=(0.9, 0.999), eps=1e-4,
        )

    def inner(self, bootstrap=False):
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        bootstrap_rollout_data = []

        # Do a complete pass on the rollout buffer
        for rollout_data in self.rollout_buffer.get(self.batch_size):

            if bootstrap == True:
                bootstrap_rollout_data.append(rollout_data)

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                self.policy.reset_noise(self.batch_size)

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage
            advantages = rollout_data.advantages
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            if self.clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip the difference between old and new value
                # NOTE: this depends on the reward scaling
                values_pred = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                )
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            #loss = self.policy_coef.item() * policy_loss + self.ent_coef.item() * entropy_loss + self.vf_coef.item() * value_loss
            loss = self.policy_coef * policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Clip grad norm
            self.policy.optimizer.step(loss)

        self.avg_rollout_reward_buffer.append(self.rollout_buffer.rewards.mean())

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        if bootstrap == True:
            return bootstrap_rollout_data

    def generate_loss(self, rollout_data, model, outer_loop=True):
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        actions = rollout_data.actions

        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            model.reset_noise(self.batch_size)

        values, log_prob, entropy = model.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        # Normalize advantage
        advantages = rollout_data.advantages
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values_pred)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -torch.mean(-log_prob)
        else:
            entropy_loss = -torch.mean(entropy)

        # PPO or BMG ?
        #self.ent_coef = self.meta_learner(torch.Tensor(self.avg_rollout_reward_buffer).to(self.meta_learner.device))
        #self.ent_coef = self.meta_learner(torch.Tensor([1.0] * 10).to(self.meta_learner.device))
        loss = self.policy_coef * policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        '''
        print()
        print("p, e, v")
        print(policy_loss)
        print(entropy_loss)
        print(value_loss)
        '''

        return loss


    def outer(self, ac_k, tb, rollout_datas, ac_k_state_dict):

        loss_tb = []
        for rollout_data in rollout_datas:
            with torch.no_grad():
                bootstrap_loss = self.generate_loss(rollout_data, tb)
                loss_tb.append(bootstrap_loss)

        torchopt.recover_state_dict(ac_k, ac_k_state_dict)

        loss_k = []
        for rollout_data in rollout_datas:
            kth_loss = self.generate_loss(rollout_data, ac_k)
            loss_k.append(kth_loss)

        l2_loss = [(loss_k[i] - loss_tb[i]) * (loss_k[i] - loss_tb[i]) for i in range(len(rollout_data))]
        l2_loss = sum(l2_loss)/len(l2_loss)

        print()
        print(self.ent_coef)
        print(self.policy_coef)
        print(self.vf_coef)

        self.coef_optimizer.zero_grad()
        l2_loss.backward()

        print(self._n_updates)

        if self.pretrain_step < self._n_updates:
            self.coef_optimizer.step()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        for _ in range(self.K):         self.inner()
        k_state_dict = torchopt.extract_state_dict(self.policy)

        for _ in range(self.L - 1):     self.inner()
        k_l_m1_state_dict = torchopt.extract_state_dict(self.policy)

        rollout_datas = self.inner(bootstrap=True)
        self.outer(self.kth_policy, self.policy, rollout_datas, k_state_dict)

        # Recover inner model parameters to the point before bootstrapping and stop gradient
        torchopt.recover_state_dict(self.policy, k_l_m1_state_dict)
        torchopt.stop_gradient(self.policy)
        torchopt.stop_gradient(self.policy.optimizer)

        # Logs
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "BMG",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            progress_bar: bool = True,
    ):
        """
        Collects rollouts at self.rollout_buffer, call self.train() and update self.num_timesteps.
        Logging and callback is also handled.
        Implemented in superclass.
        """
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            #progress_bar=progress_bar,
        )


class MLP(nn.Module):
    def __init__(self, input_size, feature_size, device):
        super(MLP, self).__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Linear(input_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, 1),
            nn.Sigmoid()
        )
        self.to(device)

    def forward(self, x):
        x = self.features(x)
        return x
