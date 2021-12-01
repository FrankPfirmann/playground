import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

seed = 1
replay_size = 1000000
batch_size = 64

gamma = 0.99
tau = 0.005
hidden_size = 256
lr_q = 0.0003
lr_policy = 0.0003

max_trans = 100000
warmup_trans = 500
eval_every = 500
eval_episodes = 10
exploration_noise = 0.3


# inspired by https://github.com/KaleabTessera/DQN-Atari/blob/master/dqn/agent.py
class DQN(object):
    def __init__(self, q_network, q_target_network, is_train=True):
        self.device = torch.device("cpu")
        torch.manual_seed(123)
        np.random.seed(123)

        # Define q and q_target
        self.q_network = q_network
        self.q_target_network = q_target_network
        self.q_optim = Adam(self.q_network.parameters(), lr=lr_q)
        self.update_target(self.q_target_network, self.q_network, 1.0)
        self.is_train = is_train

    def update_target(self, q_target, q, t):
        for x, y in zip(q_target.parameters(), q.parameters()):
            x.data.copy_(x.data * (1.0 - t) + y.data * t)

    def get_policy(self):
        def policy(obs):
            q_values = self.q_network(obs)
            if self.is_train:
                if random.random() > exploration_noise:
                    action = q_values.max(1)[1]
                else:
                    action = torch.tensor([random.randint(0, q_values.size(dim=1)-1)])
            else:
                action = q_values.max(1)[1]
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
            q_target = rwd_batch + gamma * max_next_q * (1.0 - done_batch)

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
        self.update_target(self.q_target_network, self.q_network, tau)
        return loss


