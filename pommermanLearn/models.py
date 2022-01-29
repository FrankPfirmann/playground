import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#TODO: express model structure as param
from pommerman.constants import Item

import params as p
from util.data import transform_observation, transform_observation_centralized
from util.data import merge_views
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
        self.input_dim = 256
        self.p_obs=p_obs
        self.planes_num = 13 if p_obs else 12
        self.padding = 1 if p_obs else 0
        self.use_memory = p.use_memory
        self.memory = None

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.planes_num, out_channels=self.last_cnn_depth, kernel_size=(3, 3), stride=(1, 1)),\
            nn.MaxPool2d((2,2), padding=self.padding, stride=2),\
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d((2,2),stride=2),
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
    
    def validate_memory(self, nobs):
        if self.memory is None:
            return False

        if self.memory[0].shape != nobs[0].shape:
            return False
        
        if self.memory[1].shape != nobs[1].shape:
            return False

        return True


    def update_memory(self, nobs):
        if self.memory is None or not self.validate_memory(nobs):
            self.memory = nobs
            return

        # Invert fog to get field of view
        fov = 1-nobs[0][..., 12, :, :]

        for layer in range(12):
            if layer in [0,1]: # Remember walls and passages always
                forgetfulness=0.0
            else: # Forget other layers that are out of view slowly
                forgetfulness=p.forgetfullness
            first = self.memory[0][..., layer, :, :]
            second = nobs[0][..., layer, :, :]
            merged = merge_views(first, second, fov, forgetfullness=forgetfulness)
            nobs[0][..., layer, :, :] = merged

        self.memory = nobs

    def _calc_linear_inputdim(self, board_size):
        dim = (board_size-self.conv_kernel_size+self.padding*2)/self.conv_kernel_stride + 1
        dim = np.floor((dim+ (self.pool_kernel_size -1) - 1)/self.pool_kernel_stride)

        return dim*dim*self.last_cnn_depth

    def forward(self, obs):
        if self.use_memory and self.p_obs:
            # Memory in this form only makes sense with partial
            # observability
            self.update_memory(obs)
            obs = self.memory

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

class PommerQEmbeddingMLP(nn.Module):
    def __init__(self, embedding_model, embedding_size=128):
        super(PommerQEmbeddingMLP, self).__init__()
        self.embedding_model = embedding_model

        self.linear=nn.Sequential(
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
        x_board=obs[0] # Board Embedding

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
            #flattened = torch.tensor(flattened, device=torch.device('cpu'))
            X = torch.tensor(planes, device=torch.device('cpu')).unsqueeze(0)
            board_embedding = self.embedding_model.forward(X)
            board_embedding = board_embedding.detach().numpy()
            return [
                board_embedding
            ]

        return transformer