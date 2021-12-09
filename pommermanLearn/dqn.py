import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import params as p
from util.data import transform_observation
from action_prune import get_filtered_actions

from torch.optim import Adam


# inspired by https://github.com/KaleabTessera/DQN-Atari/blob/master/dqn/agent.py

class DQN(object):
    def __init__(self, q_network, q_target_network, is_train=True):
        self.device = torch.device("cpu")
        # Define q and q_target
        self.q_network = q_network
        self.q_target_network = q_target_network
        self.q_optim = Adam(self.q_network.parameters(), lr=p.lr_q)
        self.update_target(self.q_target_network, self.q_network, 1.0)
        self.is_train = is_train

    def update_target(self, q_target, q, t):
        for x, y in zip(q_target.parameters(), q.parameters()):
            x.data.copy_(x.data * (1.0 - t) + y.data * t)

    def get_policy(self):
        def policy(obs):
            # get valid actions according to action filter and transform to be able to filter tensor
            valid_actions = get_filtered_actions(obs)
            valid_actions_transformed = []
            for i in range(6):
                valid_actions_transformed += [1] if i in valid_actions else [float("NaN")]
            valid_actions_transformed = torch.FloatTensor(valid_actions_transformed).to(torch.device("cpu")).unsqueeze(0)

            obs = transform_observation(obs)
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)

            if len(valid_actions) != 0:
                q_values = self.q_network(obs)*valid_actions_transformed
            else:
                q_values = self.q_network(obs)

            if self.is_train:
                if random.random() > p.exploration_noise:
                    action = torch.nan_to_num(q_values, nan=-float('inf')).max(1)[1]
                else:
                    action = torch.tensor([random.choice(valid_actions)])
            else:
                action = torch.nan_to_num(q_values, nan=-float('inf')).max(1)[1]
            return action
        return policy

    def set_train(self, t):
        self.is_train = t

    def update_q(self, obs, act, rwd, nobs, done):
        obs_batch = obs
        nobs_batch = nobs
        act_batch = act
        rwd_batch = rwd
        done_batch = done

        with torch.no_grad():
            next_q = self.q_target_network(nobs_batch).squeeze()
            max_next_q = next_q.max(1)[0].unsqueeze(1)
            q_target = rwd_batch + p.gamma * max_next_q * (1.0 - done_batch)

        q = self.q_network(obs_batch).squeeze()
        q = q.gather(1, act_batch.long().unsqueeze(1))
        loss = F.mse_loss(q, q_target)

        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()
        return loss.item()

    def train(self, batch):
        obs_batch, act_batch, rwd_batch, nobs_batch, done_batch = batch
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.FloatTensor(act_batch).to(self.device)
        rwd_batch = torch.FloatTensor(rwd_batch).to(self.device)
        nobs_batch = torch.FloatTensor(nobs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        loss = self.update_q(obs_batch, act_batch, rwd_batch, nobs_batch, done_batch)
        self.update_target(self.q_target_network, self.q_network, p.tau)
        return loss


