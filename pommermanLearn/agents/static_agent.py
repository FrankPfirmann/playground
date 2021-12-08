from pommerman import agents

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