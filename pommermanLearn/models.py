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
        self.memory = None

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.planes_num, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),\
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Flatten()
        )
        self.board_transform_func = board_transform_func

        self.linear=nn.Sequential(
            nn.Linear(in_features=8, out_features=6),
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

    def update_memory(self, nobs):
        if self.memory is None or not self.validate_memory(nobs):
            self.memory = [nobs[0], nobs[1]]
            return

        batch_size = nobs[0].shape[0]
        for sample in range(batch_size):
            for layer in range(12): # Merge all layers but fog
                if layer in [0,1]: # Remember walls and passages always
                    forgetfulness=0.0
                else: # Forget other layers that are out of view slowly
                    forgetfulness=p.forgetfullness

                # Invert fog to get field of view

                current_position = [nobs[1][0,1], nobs[1][0,2]]
                last_position= [self.memory[1][0,1], self.memory[1][0,2]]
                first    = self.memory[0][sample, layer, :, :]
                second   = nobs[0][sample, layer, :, :]
                fog      = nobs[0][sample, -1, :, :]
                fov      = 1 - fog

                first    = decentralize_view(first,  last_position, (11,11))
                second   = decentralize_view(second, current_position, (11,11))
                fov      = decentralize_view(fov,    current_position, (11,11))

                if p.memory_method == 'forgetting':
                    merged = merge_views_counting(first, second, fov)
                if p.memory_method == 'counting':
                    merged = merge_views(first, second, fov, forgetfullness=forgetfulness)

                if sample==0 and layer==2:
                    print(merged)

                # Recentralize and safe into memory
                merged = centralize_view(merged, current_position, 0)
                merged = torch.tensor(merged, device=p.device)
                nobs[0][sample, layer, :, :] = merged

        self.memory = nobs

    def _calc_linear_inputdim(self, board_size):
        dim = (board_size-self.conv_kernel_size+self.padding*2)/self.conv_kernel_stride + 1
        dim = np.floor((dim+ (self.pool_kernel_size -1) - 1)/self.pool_kernel_stride)

        return dim*dim*self.last_cnn_depth

    def forward(self, obs):
        if self.use_memory:
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
                    np.array(obs['message']),
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
    def __init__(self, bounds: tuple):
        self.bounds = bounds
        self.reset()

    def update(self, view: np.array, position: list) -> None:
        """
        Updates the global state with the centralized and cropped ``view``

        :param view: A cropped and centralized view of the board
        :param position: The position of the views center in the global
            coordinate system.
        """
        assert view.shape[0] == view.shape[1]

        pad_width = int((self.bounds[0]*2-1-1)/2 - (view.shape[0]-1)/2)
        fov = np.pad(np.ones(view.shape), pad_width, constant_values=0)
        view = np.pad(view, pad_width, constant_values=0)

        # Decentralize view
        fov = decentralize_view(fov, position, bounds=self.bounds)
        view = decentralize_view(view, position, bounds=self.bounds)

        # Merge views
        # TODO: Add option for merge with forgetfullness or externalize merging
        #self.merged = merge_views(self.merged, view, fov, forgetfullness=0.5)
        fog = 1-fov
        self.board = view*fov + self.board*fog+np.where(self.board != fov, fog, fov)*fog

    def get_view(self, position: list=None, view_range: int=0, centralized=False) -> np.array:
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
        if position is None:
            position = calculate_center(self.board.shape)

        view = centralize_view(self.board, position, 0)
        if view_range > 0:
            view = crop_view(view, view_range)
        
        if centralized:
            return view
        else:
            # TODO: Add unit tests
            return decentralize_view(view, position, self.bounds)
    
    def reset(self) -> None:
        """
        Reset the internal representation to its initial state
        """
        self.board = np.zeros(self.bounds) 