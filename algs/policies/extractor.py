from collections import OrderedDict

import torch
from torch import nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import TensorDict


class GsnakeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_class=None, cnn_output_dim: int = 256):
        super().__init__(observation_space, features_dim=1)
        assert cnn_class is not None, "cnn_class must be provided"

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
            encoded_tensor_list.append(extractor(grouped_observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


class MultiBoxExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=None):
        features_dim = features_dim or len(observation_space)
        super().__init__(observation_space, features_dim=features_dim)
        self.keys = sorted(observation_space.keys())
        self.extractor = nn.Sequential(*create_mlp(
            input_dim=sum([space.shape[0] for space in observation_space.values()]),
            output_dim=features_dim,
            net_arch=[64, 64],
            activation_fn=nn.ReLU,
        ))

    def forward(self, observations: TensorDict) -> torch.Tensor:
        input_ = torch.cat([observations[key] for key in self.keys], dim=1)
        return self.extractor(input_)


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
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class CustomCNN(BaseFeaturesExtractor):
    """
    Customized CNN for Gsnake performance
    :param observation_space:
    :param features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
