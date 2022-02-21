
from pommerman import agents

class TrainAgent(agents.BaseAgent):
    """
    An agent used for training a pytorch model 
    """

    def __init__(self, policy):
        """
        Initializes the agent and sets the model to train 

        :param policy: The pytorch model to train
        """
        super(TrainAgent, self).__init__()
        self.policy = policy

    def act(self, obs, action_space):
        act = self.policy(obs)
        return [int(act[0].detach().cpu().numpy()[0]), act[1], act[2]]