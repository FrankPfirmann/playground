import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#TODO: express model structure as param
from pommerman.constants import Item
from util.data import transform_observation, transform_observation_centralized
## takes in a module and applies the specified weight initialization


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.005)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.0)


class Pommer_Q(nn.Module):
    def __init__(self, p_obs, board_transform_func):
        super(Pommer_Q, self).__init__()
        self.conv_kernel_size = 3
        self.conv_kernel_stride = 1
        self.pool_kernel_size = 2
        self.pool_kernel_stride = 2
        self.last_cnn_depth = 32
        self.input_dim = 512
        self.planes_num = 13 if p_obs else 12
        self.padding = 1 if p_obs else 0

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.planes_num, out_channels=self.last_cnn_depth, kernel_size=(3, 3), stride=(1, 1)),\
            nn.MaxPool2d((2,2), padding=self.padding, stride=2),\
            # nn.Conv2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(1, 1)),
            # nn.MaxPool2d((2,2),stride=2),
            nn.Flatten()
        )
        self.board_transform_func = board_transform_func

        self.linear=nn.Sequential(
            nn.Linear(in_features=6, out_features=6),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(in_features=self.input_dim+6, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=6)
        )

        #self.conv.apply(init_weights)
        #self.linear.apply(init_weights)
        #self.combined.apply(init_weights)

    def _calc_linear_inputdim(self, board_size):
        dim = (board_size-self.conv_kernel_size+self.padding*2)/self.conv_kernel_stride + 1
        dim = np.floor((dim+ (self.pool_kernel_size -1) - 1)/self.pool_kernel_stride)

        return dim*dim*self.last_cnn_depth

    def forward(self, obs):
        x1=obs[0] # Board
        x2=obs[1] # Step, Position

        x1 = self.conv(x1)
        x2 = self.linear(x2)
        x1_2 = torch.cat((x1, x2), dim=1).squeeze()

        x = self.combined(x1_2).unsqueeze(0)
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
            return [
                self.board_transform_func(obs),
                np.array(np.hstack((
                    np.array(obs['step_count']),
                    np.array(list(obs['position'])),
                    np.array(obs['ammo']),
                    np.array(obs['can_kick']),
                    np.array(obs['blast_strength'])
                )))]

        return transformer