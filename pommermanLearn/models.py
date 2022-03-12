from ctypes import pointer
from heapq import merge
import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO: express model structure as param
from pommerman.constants import Item

import params as p
from util.data import merge_views_counting
from util.data import calculate_center
from util.data import centralize_view, decentralize_view
from util.data import transform_observation, transform_observation_centralized
from util.data import merge_views, crop_view, merge_views_life


def _init_with_value(value, board_size):
    a = np.zeros((board_size, board_size))
    a.fill(value)
    return a


class Pommer_Q(nn.Module):
    def __init__(self, board_transform_func, support=None):
        super(Pommer_Q, self).__init__()
        self.conv_out_dim = 64
        self.communicate = p.communicate
        self.planes_num = 26 if not ((p.centralize_planes and not p.p_observable) or p.crop_fog) else 25
        self.planes_num = 28 if p.communicate == 2 and p.use_memory else self.planes_num
        self.padding = 1
        self.first_hidden_out_dim = 256
        self.adv_hidden_dim = 256
        self.conv_channels = 32
        self.memory = None
        self.support = support
        self.atom_size = p.atom_size if p.categorical else 1
        self.p_obs = p.p_observable
        self.noisy_layers = []
        self.dueling = p.dueling
        self.categorical = p.categorical
        self.value_1 = self.linear_layer(self.first_hidden_out_dim, self.adv_hidden_dim, p.noisy) if self.dueling else None
        self.value_2 = self.linear_layer(self.adv_hidden_dim, self.atom_size, p.noisy) if self.dueling else None
        self.advantage_1 = self.linear_layer(self.first_hidden_out_dim, self.adv_hidden_dim, p.noisy)
        self.advantage_2 = self.linear_layer(self.adv_hidden_dim, 6 * self.atom_size, p.noisy)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.planes_num, out_channels=self.conv_channels, kernel_size=(3, 3), stride=(1, 1)), \
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.conv_channels),
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.conv_channels),
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Flatten())
        self.board_transform_func = board_transform_func

        self.hidden_linear = nn.Sequential(
            nn.Linear(in_features=288, out_features=self.first_hidden_out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.first_hidden_out_dim)
        )
        self.value_stream = nn.Sequential(
            self.value_1,
            nn.ReLU(),
            self.value_2
        )
        self.advantage_stream = nn.Sequential(
            self.advantage_1,
            nn.ReLU(),
            self.advantage_2
        )

    def linear_layer(self, in_size, out_size, noisy):
        if noisy:
            noisy_layer = NoisyLinear(in_size, out_size)
            self.noisy_layers.append(noisy_layer)
            return noisy_layer
        else:
            return nn.Linear(in_size, out_size)

    def forward(self, obs):
        if self.categorical:
            distribution = self.get_distribution(obs)
            q_values = torch.sum(distribution * self.support, dim=2)
        else:
            q_values = self.get_q_non_categorical(obs)
        return q_values

    def get_features(self, obs):
        x1 = obs[0]
        x1 = self.conv(x1)
        hidden_out = self.hidden_linear(x1)
        return hidden_out

    def get_distribution(self, obs):
        combined_out = self.get_features(obs)
        advantages = self.advantage_stream(combined_out)
        advantages_view = advantages.view(-1, 6, self.atom_size)
        if self.dueling:
            values = self.value_stream(combined_out)
            values_view = values.view(-1, 1, self.atom_size)
            q_atoms = values_view + advantages_view - advantages_view.mean(dim=1, keepdim=True)
            dist = F.softmax(q_atoms, dim=-1)
        else:
            dist = F.softmax(advantages_view, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist

    def get_q_non_categorical(self, obs):
        combined_out = self.get_features(obs)
        advantages = self.advantage_stream(combined_out)
        if self.dueling:
            values = self.value_stream(combined_out)
            q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        else:
            q_values = advantages
        return q_values

    def reset_noise(self):
        for layer in self.noisy_layers:
            layer.reset_noise()

    def get_transformer(self) -> Callable:
        """
        Return a callable for input transformation.

        The callable should take a ``dict`` containing data of a single
        observation from the Pommerman environment and return a ``list``
        of individual numpy arrays that can be used later as an input
        value in the ``forward()`` function.
        """
        def transformer(obs: dict, pre_transformed=None) -> list:
            enemy_dead= [0 if e.value in obs['alive'] else 1 for e in obs['enemies'][:2]]
            teammate_v = obs['teammate'].value
            self_v = ((teammate_v -8) % 4)+ 10
            teammate_dead = 0 if teammate_v in obs['alive'] else 1
            self_dead = 0 if self_v in obs['alive'] else 1

            board = pre_transformed if pre_transformed is not None else self.board_transform_func(obs)
            board_size = board.shape[1]

            step_count = obs['step_count'] if not p.normalize_steps else obs['step_count']/p.max_steps
            fl = [_init_with_value(step_count, board_size),
                  _init_with_value(obs['position'][0], board_size),
                  _init_with_value(obs['position'][1], board_size),
                  _init_with_value(obs['ammo'], board_size),
                  _init_with_value(obs['can_kick'], board_size),
                  _init_with_value(obs['blast_strength'], board_size),
                  _init_with_value(float(self_dead), board_size),
                  _init_with_value(float(enemy_dead[0]), board_size),
                  _init_with_value(float(enemy_dead[1]), board_size),
                  _init_with_value(float(teammate_dead), board_size),
                  ]

            if self.communicate == 2:
                fl += [_init_with_value(obs['message'][0], board_size),
                       _init_with_value(obs['message'][1], board_size)]

            fl = np.array(fl)
            board = np.vstack((board, fl))
            return [board]

        return transformer


class PommerQEmbeddingMLP(nn.Module):
    def __init__(self, embedding_model, embedding_size=128):
        super(PommerQEmbeddingMLP, self).__init__()
        self.embedding_model = embedding_model

        self.linear = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=6),
            nn.Softmax(dim=2)
        )

    def forward(self, obs):
        x_board = obs[0]  # Board Embedding

        x = self.linear(x_board).squeeze()
        return x

    def get_transformer(self) -> Callable:
        """
        Return a callable for input transformation.

        The callable should take a ``dict`` containing data of a single
        observation from the Pommerman environment and return a ``list``
        of individual numpy arrays that can be used later as an input
        value in the ``forward()`` function.
        """

        def transformer(obs: dict) -> list:
            planes = transform_observation(obs, p_obs=True, centralized=True)
            planes = np.array(planes, dtype=np.float32)

            # TODO: Make 'cpu' variable
            # Generate embedding
            # flattened = torch.tensor(flattened, device=torch.device('cpu'))
            X = torch.tensor(planes, device=torch.device('cpu')).unsqueeze(0)
            board_embedding = self.embedding_model.forward(X)
            board_embedding = board_embedding.detach().numpy()
            return [
                board_embedding
            ]

        return transformer


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
        https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        sigma_zero (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, sigma_zero: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_zero = sigma_zero

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.initialize_weights()
        self.reset_noise()

    def initialize_weights(self):
        """Initializing the trainable network parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.sigma_zero / np.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.sigma_zero / np.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Sample new noise for the network"""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # factorized gaussian noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    def scale_noise(self, size: int) -> torch.Tensor:
        """Scale the noise with real-valued function"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class PommerQEmbeddingRNN(nn.Module):
    def __init__(self, embedding_model):
        super(PommerQEmbeddingRNN, self).__init__()
        self.embedding_model = embedding_model
        self.memory = []
        self.steps = 10

        # Stacked lstm
        self.rnn = [nn.LSTM(64, 64) for step in range(self.steps)]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=6),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        while len(self.memory) >= self.steps:
            self.memory.pop(0)

        while len(self.memory) != self.steps:
            self.memory.append(obs)

        # x=obs[0] # Board Embedding

        x = None
        h = None
        for obs_n, rnn_n in zip(self.memory, self.rnn):
            x_n = obs_n[0]
            x, h = rnn_n(x_n, h)

        x = self.linear(x).squeeze()
        return x

    def get_transformer(self) -> Callable:
        """
        Return a callable for input transformation.

        The callable should take a ``dict`` containing data of a single
        observation from the Pommerman environment and return a ``list``
        of individual numpy arrays that can be used later as an input
        value in the ``forward()`` function.
        """

        def transformer(obs: dict) -> list:
            planes = transform_observation(obs, p_obs=True, centralized=True)
            planes = np.array(planes, dtype=np.float32)

            # Generate embedding
            # flattened = planes.flatten()
            # flattened = torch.tensor(flattened, device=torch.device('cpu')) # TODO: Make 'cpu' variable
            X = torch.tensor(planes, device=torch.device('cpu')).unsqueeze(0)
            board_embedding = self.embedding_model.forward(X)
            board_embedding = board_embedding.detach().numpy()
            return [
                board_embedding
            ]

        return transformer
