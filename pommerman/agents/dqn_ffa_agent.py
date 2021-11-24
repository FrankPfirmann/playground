from pommerman.agents.base_agent import BaseAgent


class DQNFFAAgent(BaseAgent):
    """
    Agent for the FFA method using a neural network model e.g., CNN
    """

    def __init__(self, nn_model):
        # TODO:init parameters
        self.nn_model = nn_model

    def act(self, obs, action_space):
        # TODO:act based on NN Model and perhaps store state values(which?)
        return action_space.sample()

    def episode_end(self, reward):
        # storing rewards should be done by game_player
        return
