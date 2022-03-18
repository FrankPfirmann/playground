import torch
from pommerman.agents import BaseAgent
from agents.train_agent import TrainAgent
from models.pommer_q import Pommer_Q
from dqn import DQN
from util.data import transform_observation_partial, transform_observation_partial_uncropped
import params as p

class DQNAgent(TrainAgent):
    """
    A DQN based agent for the Pommerman environment
    """
    def __init__(self, model_dir):
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

        super(DQNAgent, self).__init__(policy=dqn.get_policy(), transformer=transformer, is_train=False)