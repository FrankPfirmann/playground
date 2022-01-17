import torch
from pommerman.runner import DockerAgentRunner
from agents.train_agent import TrainAgent
from models import Pommer_Q
from dqn import DQN
from util.data import transform_observation_partial

class DockerAgent(DockerAgentRunner):
    """
    An agent that exposes a REST API for usage in a docker container.
    """
    def __init__(self):
        model_dir="./data/models/deployment.pkl"

        q = Pommer_Q(11*2-1, transform_observation_partial)
        q_target = Pommer_Q(11*2-1, transform_observation_partial)

        algo = DQN(q, q_target, is_train=False, device=torch.device("gpu"))
        algo.q_network.load_state_dict(torch.load(model_dir))

        self._agent=TrainAgent(policy=algo.get_policy())

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()