
from pommerman import agents

class TrainAgent(agents.BaseAgent):
    """
    An agent used for training a pytorch model 
    """

    def __init__(self, policy, algo="dqn"):
        """
        Initializes the agent and sets the model to train 

        :param policy: The pytorch model to train
        """
        super(TrainAgent, self).__init__()
        self.policy = policy
        self.algo = algo

    def act(self, obs, action_space):

        if self.algo == "dqn":
            act = self.policy(obs)
            act = int(act.detach().cpu().numpy()[0])
            
            return act
        elif self.algo == "ppo":
            act, act_logprob, obs = self.policy(obs)

            return act, act_logprob, obs