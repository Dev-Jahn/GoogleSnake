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
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.distributions import kl_divergence as kld


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
            # Base on-policy algorithm
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-1,
            n_steps: int = 5,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            normalize_advantage: bool = True,
            vf_coef: float = 0.4,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            # Meta-learning related parameters
            K: int = 1,
            L: int = 1,
            meta_window_size: int = 10,
            meta_learning_rate: Union[float, Schedule] = 1e-4,
            # PPO
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            target_kl: Optional[float] = None,
            # misc
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0,
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
        self.before_bootstrap = None

        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

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
            self.policy.parameters(),
            lr=self.meta_learning_rate, betas=(0.9, 0.999), eps=1e-4,
        )

    def inner(self, bootstrap=False):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        param_grads = dict()

        # If batch_size=None, iterate with n_envs*n_steps total samples(steps)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            self.avg_rollout_reward_buffer.append(self.rollout_buffer.rewards.mean())
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # predict entropy coefficient with meta-learner
            ent_coef = self.meta_learner(torch.Tensor(self.avg_rollout_reward_buffer).to(self.meta_learner.device))

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            if bootstrap:
                loss = policy_loss
            else:
                # l2norm = sum([p.norm() for p in self.policy.mlp_extractor.value_net.parameters()])
                # l2norm = sum([p.norm() for p in self.policy.value_net.parameters()])
                # loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss + l2norm
                loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss

            for key, param in dict(self.policy.named_parameters()).items():
                with torch.no_grad():
                    param_grads.update({key: param})
            # TODO: check if grad clipping is compatible with meta-learning
            # Clip grad norm
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            # Optimization step
            self.policy.optimizer.step(loss)

        # For Gradient Monitoring
        # print(f'{self._n_updates + 1} inner updates')
        # print(f'{(self._n_updates + 1) % (self.K + self.L)} / K+L({self.K}+{self.L})')
        # for key, param in dict(self.policy.named_parameters()).items():
        #     with torch.no_grad():
        #         grad_norm = (param - param_grads[key]).norm().item()
        #         print(f'{key:35}| Grad Norm | {grad_norm:.6f}')
        # print('=' * 80)

        # Logging
        # if len(self.ep_info_buffer) ==0:
        #     self.logger.record("rollout/ep_food_taken_mean", 0)
        # else:
        #     self.logger.record(
        #         "rollout/ep_food_taken_mean", np.mean([info['food_taken'] for info in self.ep_info_buffer]))
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/inner_iter", self._n_updates)
        self.logger.record("train/outer_iter", self._n_updates // (self.K * self.L))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        # Checkpoint when K-th inner loop is done
        if self._n_updates % (self.K + self.L) == self.K:
            torchopt.recover_state_dict(self.kth_policy, torchopt.extract_state_dict(self.policy))

        if self._n_updates % (self.K + self.L) == (self.K + self.L - 1):
            self.before_bootstrap = torchopt.extract_state_dict(self.policy)

    def outer(self):
        rollout_data = next(self.rollout_buffer.get())
        kth_dist = self.kth_policy.get_distribution(rollout_data.observations)
        with torch.no_grad():
            bootstrap_dist = self.policy.get_distribution(rollout_data.observations)
        loss = self._matching_function(kth_dist, bootstrap_dist)
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

    def _matching_function(self, kth_dist, bootstrapped_dist):
        return kld(kth_dist, bootstrapped_dist).sum()

    def train(self) -> None:
        if self._n_updates % (self.K + self.L) < (self.K + self.L - 1):
            self.inner()
        else:  # At (K+L)th iteration of cycle, do meta-learning
            self.inner(bootstrap=True)
            self.outer()
            # Recover inner model parameters to the point before bootstrapping and stop gradient
            torchopt.recover_state_dict(self.policy, self.before_bootstrap)
            torchopt.stop_gradient(self.policy)
            torchopt.stop_gradient(self.policy.optimizer)

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
            progress_bar=progress_bar,
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
        return self.features(x)
