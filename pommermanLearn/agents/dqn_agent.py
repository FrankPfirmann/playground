import torch
from pommerman.agents import BaseAgent
from agents.train_agent import TrainAgent
from models.pommer_q import Pommer_Q
from dqn import DQN
from util.data import transform_observation_partial, transform_observation_partial_uncropped
import params as p

class DQNAgent(BaseAgent):
    """
    A DQN based agent for the Pommerman environment
    """
    def __init__(self, model_dir):
        super(DQNAgent, self).__init__()
        support = torch.linspace(p.v_min, p.v_max, p.atom_size).to(p.device)
        transformer = transform_observation_partial_uncropped
        q = Pommer_Q(transform_observation_partial, support)
        q.load_state_dict(torch.load(model_dir, map_location=p.device))
        q.to(torch.device(p.device))
        q.eval()
        q_target = Pommer_Q(transform_observation_partial, support)
        q_target.load_state_dict(torch.load(model_dir, map_location=p.device))
        q_target.to(torch.device(p.device))
        q_target.eval()
        dqn = DQN(q, q_target, p.exploration_noise, device=torch.device(p.device), dq=p.double_q, support=support)

        transformer = q.get_transformer()
        self._agent=TrainAgent(policy=dqn.get_policy(), transformer=transformer, is_train=False)

    def init_agent(self, id, game_type):
        super(DQNAgent, self).init_agent(id, game_type)
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()