'''An agent that preforms a random action each step'''
from pommerman.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        return action_space.sample()
