from ctypes import pointer
from heapq import merge
import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#TODO: express model structure as param
from pommerman.constants import Item

import params as p
from util.data import merge_views_counting
from util.data import calculate_center
from util.data import centralize_view, decentralize_view
from util.data import transform_observation, transform_observation_centralized
from util.data import merge_views, crop_view
## takes in a module and applies the specified weight initialization


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.005)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.0)


class Pommer_Q(nn.Module):
    def __init__(self, p_central, board_transform_func):
        super(Pommer_Q, self).__init__()
        self.conv_kernel_size = 3
        self.conv_kernel_stride = 1
        self.pool_kernel_size = 2
        self.pool_kernel_stride = 2
        self.last_cnn_depth = 32
        self.input_dim = 64
        self.planes_num = 15 if p_central else 14
        self.padding = 1
        self.use_memory = False
        self.use_memory = p.use_memory
        self.tracker = BoardTracker()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.planes_num, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),\
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Flatten()
        )
        self.board_transform_func = board_transform_func

        self.linear=nn.Sequential(
            nn.Linear(in_features=6, out_features=6),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.LazyLinear(out_features=32),
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
        
        if self.memory[1][0,0]+1 != nobs[1][0,0]:
            return False

        return True

    def _calc_linear_inputdim(self, board_size):
        dim = (board_size-self.conv_kernel_size+self.padding*2)/self.conv_kernel_stride + 1
        dim = np.floor((dim+ (self.pool_kernel_size -1) - 1)/self.pool_kernel_stride)

        return dim*dim*self.last_cnn_depth

    def forward(self, obs):
        if self.use_memory:
            position = [obs[1][0,1], obs[1][0,2]]
            self.tracker.update(obs[0], position)
            obs[0] = self.tracker.get_view(position, view_range=4, centralized=True)

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

class PommerQEmbeddingRNN(nn.Module):
    def __init__(self, embedding_model):
        super(PommerQEmbeddingRNN, self).__init__()
        self.embedding_model = embedding_model
        self.memory=[]
        self.steps = 10

        # Stacked lstm
        self.rnn = [nn.LSTM(64, 64) for step in range(self.steps)]

        self.linear=nn.Sequential(
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

        #x=obs[0] # Board Embedding

        x = None
        h = None
        for obs_n, rnn_n in zip(self.memory, self.rnn):
            x_n=obs_n[0]
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
            #flattened = planes.flatten()
            #flattened = torch.tensor(flattened, device=torch.device('cpu')) # TODO: Make 'cpu' variable
            X = torch.tensor(planes, device=torch.device('cpu')).unsqueeze(0)
            board_embedding = self.embedding_model.forward(X)
            board_embedding = board_embedding.detach().numpy()
            return [
                board_embedding
            ]

        return transformer

class BoardTracker:
    def __init__(self):
        self.reset()

    def update(self, view: np.array, position: list) -> None:
        """
        Updates the global state with the centralized and cropped ``view``

        :param view: A cropped and centralized view of the board
        :param position: The position of the views center in the global
            coordinate system.
        """
        assert view.shape[-1] == view.shape[-2]

        if self.board is None or self.board.shape != view.shape :
            self.position = position
            self.board = view
            return

        batch_size = view.shape[0]
        for sample in range(batch_size):
            for layer in range(12): # Merge all layers but fog
                if layer in [0,1]: # Remember walls and passages always
                    forgetfulness=0.0
                else: # Forget other layers that are out of view slowly
                    forgetfulness=p.forgetfullness

                # Invert fog to get field of view

                first    = self.board[sample, layer, :, :]
                second   = view[sample, layer, :, :]
                fog      = view[sample, -1, :, :]
                fov      = 1 - fog

                first    = decentralize_view(first,  self.position, (11,11))
                second   = decentralize_view(second, position, (11,11))
                fov      = decentralize_view(fov,    position, (11,11))

                if p.memory_method == 'forgetting':
                    merged = merge_views(first, second, fov, forgetfullness=forgetfulness)
                elif p.memory_method == 'counting':
                    merged = merge_views_counting(first, second, fov)

                # Recentralize and safe into memory
                self.board[sample, layer, :, :] = centralize_view(merged, position)
        self.position = position


    def get_view(self, position: list, view_range: int=0, centralized=False) -> np.array:
        """
        Return a view of the internal global board state

        :param position: The position where the view center should lie
        :param view_range: The amount of tiles left in each direction.
            Everything outside will be cropped out if the value is
            greater then zero.   
        :param centralized: Center the returned view if True, overwise
            return an uncentered board.

        :return: The resulting view
        """
        batch_size = self.board.shape[0]
        layers     = self.board.shape[1]

        if view_range > 0:
            fov    = 2*view_range + 1
            bounds = (batch_size, layers, fov, fov)
        else:
            bounds = self.board.shape
        
        if torch.is_tensor(self.board):
            views = torch.zeros(bounds, device=p.device)
        else:
            views = np.zeros(bounds)

        for sample in range(batch_size):
            for layer in range(layers):
                #view = centralize_view(self.board[sample, layer, :, :], position, 0)
                view = self.board[sample, layer, :, :]
                if view_range > 0:
                    view = crop_view(view, view_range)
                if not centralized:
                    view = decentralize_view(view, position, self.bounds)
                
                views[sample, layer, :, :] = view
        return views
    
    def reset(self) -> None:
        """
        Reset the internal representation
        """
        self.board = None
        self.position = None