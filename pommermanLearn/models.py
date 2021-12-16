import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#TODO: express model structure as param

from util.data import transform_observation

class DQN_Q(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size):
        super(DQN_Q, self).__init__()

        self.linear1 = nn.Linear(obs_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, act_size)

    def forward(self, obs):
        x1 = F.relu(self.linear1(obs))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1


class Pommer_Q(nn.Module):
    def __init__(self, board_size):
        super(Pommer_Q, self).__init__()
        self.conv_kernel_size = 3
        self.conv_kernel_stride = 1
        self.padding = 0
        self.pool_kernel_size = 2
        self.pool_kernel_stride = 2
        self.last_cnn_depth = 32
        self.input_dim= int(self._calc_linear_inputdim(board_size))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=self.last_cnn_depth, kernel_size=(3, 3), stride=(1, 1)),\
            nn.MaxPool2d((2,2),stride=2),\
            # nn.Conv2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(1, 1)),
            # nn.MaxPool2d((2,2),stride=2),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=6),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=6)
        )

    def _calc_linear_inputdim(self, board_size):
        dim = (board_size-self.conv_kernel_size+self.padding*2)/self.conv_kernel_stride + 1
        dim = np.floor((dim+ (self.pool_kernel_size -1) - 1)/self.pool_kernel_stride)

        return dim*dim*self.last_cnn_depth

    def forward(self, obs):
        x = self.conv(obs).squeeze()
        x = self.linear(x).unsqueeze(0)
        return x

    def get_transformer(self) -> Callable:
        """
        Return a callable to transform a single observation from the
        Pommerman environment to an input format supported by the model.
        """
        def transformer(obs: dict) -> np.array:
            return transform_observation(obs)

        return transformer