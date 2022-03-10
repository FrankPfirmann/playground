import random
from pommerman import agents

from pommerman.constants import Action
import action_prune


class StaticAgent(agents.BaseAgent):
    """
    An agent that returns always the same predifined action
    """

    def __init__(self, action):
        """
        Initializes the agent with the action to always return

        :param action: The action to return always by `act()`
        """
        super(StaticAgent, self).__init__()
        self.action = action

    def act(self, obs, action_space):
        return self.action


class StaticAgentNoBomb(agents.BaseAgent):
    """
    An agent that returns always the same predifined action
    """

    def __init__(self, action):
        """
        Initializes the agent with the action to always return

        :param action: The action to return always by `act()`
        """
        super(StaticAgent, self).__init__()
        self.action = action

    def act(self, obs, action_space):
        valid_actions = action_prune.get_filtered_actions(obs)
        if Action.Stop.value in valid_actions or len(valid_actions) == 0:
            return Action.Stop.value
        else:
            return random.choice(valid_actions)
