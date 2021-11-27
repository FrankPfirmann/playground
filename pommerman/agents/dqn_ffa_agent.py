import numpy as np

import pommerman.characters
from pommerman.agents.base_agent import BaseAgent


class DQNFFAAgent(BaseAgent):
    """
    Agent for the FFA method using a neural network model e.g., CNN
    """

    def __init__(self, nn_model, character=pommerman.characters.Bomber):
        # TODO:init parameters
        super(DQNFFAAgent, self).__init__()
        self.nn_model = nn_model
        self.state = None

    def act(self, obs, action_space):
        # TODO:act based on NN Model and perhaps store state values(which?)
        def _get_binary_board(index, board_w, players=False):
            if players:
                t = np.where(board_w >= index, board_w, 0.)
                return np.where(t < index, t, 1.)
            t = np.where(board_w == index, board_w, 0.)
            return np.where(t != index, t, 1.)

        # create bitmaps for position, different tiles, and other agents(10)
        def _transform_board(board_obs):
            # make 0 the default instead of passage indicator
            pos = np.zeros_like(board_obs.astype(np.float32))
            pos[obs["position"][0]][obs["position"][1]] = 1
            board_w = np.where(board_obs != 0, board_obs, -1)
            return pos, _get_binary_board(-1, board_w), _get_binary_board(1, board_w), _get_binary_board(2, board_w), \
                   _get_binary_board(3, board_w), _get_binary_board(4, board_w), _get_binary_board(10, board_w, True)

        boards = np.array(_transform_board(obs["board"]))
        self.state = boards
        return action_space.sample()

    def episode_end(self, reward):
        # storing rewards should be done by game_player
        return
