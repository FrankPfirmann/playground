from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.noisy_linear import NoisyLinear
import params as p

class Pommer_Q(nn.Module):
    def __init__(self, board_transform_func, support=None):
        super(Pommer_Q, self).__init__()
        self.communicate = p.communicate
        self.planes_num = 26 if not ((p.centralize_planes and not p.p_observable) or p.crop_fog) else 25
        self.planes_num = 28 if p.communicate == 2 and p.use_memory else self.planes_num
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
            fl = [self._init_with_value(step_count, board_size),
                  self._init_with_value(obs['position'][0], board_size),
                  self._init_with_value(obs['position'][1], board_size),
                  self._init_with_value(obs['ammo'], board_size),
                  self._init_with_value(obs['can_kick'], board_size),
                  self._init_with_value(obs['blast_strength'], board_size),
                  self._init_with_value(float(self_dead), board_size),
                  self._init_with_value(float(enemy_dead[0]), board_size),
                  self._init_with_value(float(enemy_dead[1]), board_size),
                  self._init_with_value(float(teammate_dead), board_size),
                  ]

            if self.communicate == 2:
                fl += [self._init_with_value(obs['message'][0], board_size),
                       self._init_with_value(obs['message'][1], board_size)]

            fl = np.array(fl)
            board = np.vstack((board, fl))
            return [board]

        return transformer

    def _init_with_value(self, value, board_size):
        a = np.zeros((board_size, board_size))
        a.fill(value)
        return a




