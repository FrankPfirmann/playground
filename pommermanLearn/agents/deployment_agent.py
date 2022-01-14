import torch
from pommerman import agents
from pommerman.runner import DockerAgentRunner
from agents.train_agent import TrainAgent
from models import Pommer_Q
from dqn import DQN
from util.data import transform_observation_partial

from pommerman.agents.base_agent import BaseAgent

class DeploymentAgent(BaseAgent, DockerAgentRunner):
    """
    An agent used for training a pytorch model 
    """

    def __init__(self):
        self.policy = None
        pass

    def init_agent(self, id, game_type):
        """
        Initializes the agent and sets the model to train 

        :param policy: The pytorch model to train
        """
        super(DeploymentAgent, self).__init__()
        self.device = torch.device("cpu")
        model_dir="./data/models/deployment.pkl"

        q = Pommer_Q(11*2-1, transform_observation_partial)
        q_target = Pommer_Q(11*2-1, transform_observation_partial)

        algo = DQN(q, q_target)
        algo.q_network.load_state_dict(torch.load(model_dir))
        algo.set_train(False)
        self.policy = algo.get_policy()

        #self.agent=TrainAgent(policy=self.policy)
	
    def act(self, obs, action_space):
        act = self.policy(obs)
        return int(act.detach().numpy()[0])