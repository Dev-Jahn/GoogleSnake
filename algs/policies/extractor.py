from collections import OrderedDict

from functools import partial

import torch
import torchopt
from torch import nn
import gym
from stable_baselines3.common.distributions import *
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp, CombinedExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import MultiInputActorCriticPolicy as MultiInputActorCriticPolicy_
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union


class GsnakeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_class=None, cnn_output_dim: int = 256):
        super().__init__(observation_space, features_dim=1)
        assert cnn_class is not None, "cnn_class must be provided"

        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = OrderedDict()

        total_concat_size = 0
        extractors['grid'] = cnn_class(observation_space['grid'], features_dim=cnn_output_dim)
        total_concat_size += cnn_output_dim
        head_dict = gym.spaces.Dict(dict(filter(lambda kv: kv[0].startswith('head'), observation_space.items())))
        extractors['head'] = MultiBoxExtractor(head_dict)
        total_concat_size += len(head_dict)
        tail_dict = gym.spaces.Dict(dict(filter(lambda kv: kv[0].startswith('tail'), observation_space.items())))
        extractors['tail'] = MultiBoxExtractor(tail_dict)
        total_concat_size += len(tail_dict)
        portal_dict = gym.spaces.Dict(dict(filter(lambda kv: kv[0].startswith('portal'), observation_space.items())))
        extractors['portal'] = MultiBoxExtractor(portal_dict)
        total_concat_size += len(portal_dict)

        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:

        encoded_tensor_list = []
        grouped_observations = {
            'grid': observations['grid'],
            'head': dict(filter(lambda kv: kv[0].startswith('head'), observations.items())),
            'tail': dict(filter(lambda kv: kv[0].startswith('tail'), observations.items())),
            'portal': dict(filter(lambda kv: kv[0].startswith('portal'), observations.items())),
        }
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


class MultiBoxExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        features_dim = features_dim or len(observation_space)
        super().__init__(observation_space, features_dim=features_dim)
        self.keys = sorted(observation_space.keys())
        self.extractor = nn.Sequential(*create_mlp(
            input_dim=sum([space.shape[0] for space in observation_space.values()]),
            output_dim=features_dim,
            net_arch=[64, 64],
            activation_fn=nn.ReLU,
        )).to(self.device)

    def forward(self, observations: TensorDict) -> torch.Tensor:
        input_ = torch.cat([observations[key] for key in self.keys], dim=1)
        return self.extractor(input_.to(device))


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ).to(self.device)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU()).to(self.device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations).to(self.device))



class MultiInputActorCriticPolicy(MultiInputActorCriticPolicy_):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space (Tuple)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Uses the CombinedExtractor
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist,
                        (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        torchopt.accelerated_op_available(torch.device('cuda'))
        self.optimizer = self.optimizer_class(self, lr=lr_schedule(1), moment_requires_grad=True, **self.optimizer_kwargs)
        # if MetaAdam
        #self.optimizer = self.optimizer_class(self, lr=lr_schedule(1), moment_requires_grad=True, use_accelerated_op=True, **self.optimizer_kwargs)

