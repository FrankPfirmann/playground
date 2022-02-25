import torch
from pommerman.runner import DockerAgentRunner
from agents.train_agent import TrainAgent
from models import Pommer_Q
from dqn import DQN
from util.data import transform_observation_partial, transform_observation_partial_uncropped
import params as p

class DockerAgent(DockerAgentRunner):
    """
    An agent that exposes a REST API for usage in a docker container.
    """
    def __init__(self):
        #model_dir="./data/models/20220210T204437-149_1" # Agent 1
        model_dir="./data/models/20220210T204437-149_2" # Agent 2

        support = torch.linspace(p.v_min, p.v_max, p.atom_size).to(p.device)
        #device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        transformer = transform_observation_partial_uncropped
        q = Pommer_Q(True, transform_observation_partial, support)
        q.load_state_dict(torch.load(model_dir, map_location=device))
        q.to(torch.device(device))
        q_target = Pommer_Q(True, transform_observation_partial, support)
        q_target.load_state_dict(torch.load(model_dir, map_location=device))
        q_target.to(torch.device(device))
        dqn = DQN(q, q_target, p.exploration_noise, device=torch.device(device), dq=p.double_q, support=support)

        transformer = q.get_transformer()
        self._agent=TrainAgent(policy=dqn.get_policy(), transformer=transformer, is_train=False)

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()