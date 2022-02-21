import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from action_prune import get_filtered_actions
from message_util import get_message
from models import Pommer_Q
import params as p



# inspired by https://github.com/KaleabTessera/DQN-Atari/blob/master/dqn/agent.py

class DQN(object):
    def __init__(self, q_network: Pommer_Q, q_target_network: Pommer_Q, init_exploration,
                 is_train: bool=True, device: torch.device=None, dq:bool=False):
        """
        Create a new DQN model for training or inference.

        :param q_network: The train network
        :param q_target_network: The stable network
        :param is_train: If True, random exploration is enabled
        :param device: If set, use the given device. Otherwise use the CPU.
        """
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # Define q and q_target
        self.q_network = q_network
        self.q_target_network = q_target_network
        self.q_optim = Adam(self.q_network.parameters(), lr=p.lr_q)
        self.update_target(self.q_target_network, self.q_network, 1.0)
        self.is_train = is_train
        self.exploration = init_exploration
        self.double_q = dq

    def update_target(self, q_target, q, t):
        for x, y in zip(q_target.parameters(), q.parameters()):
            x.data.copy_(x.data * (1.0 - t) + y.data * t)

    def get_policy(self):
        def policy(obs):
            # get valid actions according to action filter and transform to be able to filter tensor
            valid_actions = get_filtered_actions(obs)
            message = get_message(obs)
            valid_actions_transformed = []
            for i in range(6):
                valid_actions_transformed += [1] if i in valid_actions else [float("NaN")]
            valid_actions_transformed = torch.FloatTensor(valid_actions_transformed).to(self.device).unsqueeze(0)

            obs = self.q_network.get_transformer()(obs)
            obs = [torch.FloatTensor(o).to(self.device).unsqueeze(0) for o in obs]

            if len(valid_actions) != 0:
                q_values = self.q_network(obs)*valid_actions_transformed
            else:
                q_values = self.q_network(obs)

            if self.is_train:
                if random.random() > self.exploration:
                    action = torch.nan_to_num(q_values, nan=-float('inf')).max(1)[1]
                else:
                    action = torch.tensor([random.choice(valid_actions)])
            else:
                action = torch.nan_to_num(q_values, nan=-float('inf')).max(1)[1]
            return [action] + message
        return policy

    def set_train(self, t):
        self.is_train = t

    def set_exploration(self, exp):
        self.exploration = exp

    def update_q(self, obs, act, rwd, nobs, done, weights=None):
        obs_batch = obs
        nobs_batch = nobs
        act_batch = act
        rwd_batch = rwd
        done_batch = done

        q = self.q_network(obs_batch).squeeze()
        q = q.gather(1, act_batch.long().unsqueeze(1))
        with torch.no_grad():
            if self.double_q:
                next_q = self.q_network(nobs_batch).squeeze()
                max_action_q = next_q.argmax(1)
                next_q_target = self.q_target_network(nobs_batch).squeeze()
                max_next_q = next_q_target.gather(1, max_action_q.long().unsqueeze(1))
                q_target = rwd_batch + p.gamma * max_next_q * (1.0 - done_batch)
            else:
                next_q = self.q_target_network(nobs_batch).squeeze()
                max_next_q = next_q.max(1)[0].unsqueeze(1)
                q_target = rwd_batch + p.gamma * max_next_q * (1.0 - done_batch)
            # get td error for prioritized exp replay
            td_error = q - q_target
        loss = F.mse_loss(q_target, q)
        if p.prioritized_replay:
            loss = F.mse_loss(q_target, q, reduction='none')
            weights = torch.tensor(weights).to(self.device)
            loss = torch.mean(weights * loss)
        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()
        return loss.item(), td_error


    def train(self, batch):
        if p.prioritized_replay:
            obs_batch, act_batch, rwd_batch, nobs_batch, done_batch, weights, indexes = batch
        else:
            obs_batch, act_batch, rwd_batch, nobs_batch, done_batch = batch
            weights = None
            indexes = None
        obs_batch = [np.array(obs) for obs in list(zip(*obs_batch))]
        obs_batch = [torch.FloatTensor(obs).to(self.device) for obs in obs_batch]
        act_batch = torch.FloatTensor(act_batch).to(self.device)
        rwd_batch = torch.FloatTensor(rwd_batch).to(self.device)
        nobs_batch = [np.array(obs) for obs in list(zip(*nobs_batch))]
        nobs_batch = [torch.FloatTensor(nobs).to(self.device) for nobs in nobs_batch]
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        loss, td_error = self.update_q(obs_batch, act_batch, rwd_batch, nobs_batch, done_batch, weights)
        self.update_target(self.q_target_network, self.q_network, p.tau)
        return loss, indexes, td_error

