import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from action_prune import get_filtered_actions
from models import Pommer_Q
import params as p



# inspired by https://github.com/KaleabTessera/DQN-Atari/blob/master/dqn/agent.py

class DQN(object):
    def __init__(self, q_network: Pommer_Q, q_target_network: Pommer_Q, init_exploration,
                 is_train: bool=True, device: torch.device=None, dq:bool=p.double_q, support=None):
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
        self.v_max = p.v_max
        self.v_min = p.v_min
        self.atom_size = p.atom_size
        self.batch_size = p.batch_size
        self.gamma = p.gamma
        self.support = support
        self.categorical = p.categorical
        self.use_n_step = p.use_nstep
        self.n_step = p.nsteps

    def update_target(self, q_target, q, t):
        for x, y in zip(q_target.parameters(), q.parameters()):
            x.data.copy_(x.data * (1.0 - t) + y.data * t)

    def get_policy(self):
        def policy(obs, valid_actions):
            # get valid actions according to action filter and transform to be able to filter tensor
            valid_actions_transformed = []
            for i in range(6):
                valid_actions_transformed += [1] if i in valid_actions else [float("NaN")]
            valid_actions_transformed = torch.FloatTensor(valid_actions_transformed).to(self.device).unsqueeze(0)

            #obs = self.q_network.get_transformer()(obs)
            obs = [torch.FloatTensor(o).to(self.device).unsqueeze(0) for o in obs]

            self.q_network.reset_noise()
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
            return action
        return policy

    def set_train(self, t):
        self.is_train = t

    def set_exploration(self, exp):
        self.exploration = exp

    def update_q(self, obs, act, rwd, nobs, done, weights, indices):
        loss_per_batch_elem = self.calculate_loss(obs, act, rwd, nobs, done, self.categorical)

        weights = torch.tensor(weights).to(self.device)
        loss = torch.mean(loss_per_batch_elem * weights)
        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()
        return loss.item(), loss_per_batch_elem

    def calculate_loss(self, obs_batch, act_batch, rwd_batch, nobs_batch, done_batch, categorical):
        if categorical:
            return self.calculate_loss_categorical(obs_batch, act_batch, rwd_batch, nobs_batch, done_batch)
        else:
            return self.calculate_loss_no_categorical(obs_batch, act_batch, rwd_batch, nobs_batch, done_batch)

    def update_q_with_n(self, obs, act, rwd, nobs, done, weights, indices,
                 obs_batch_n, act_batch_n, rwd_batch_n, nobs_batch_n, done_batch_n):
        loss_per_batch_elem = self.calculate_loss(obs, act, rwd, nobs, done, self.categorical)

        weights = torch.tensor(weights).to(self.device)
        # combined 1-step and n-loss

        loss_per_batch_elem_n = self.calculate_loss(obs_batch_n, act_batch_n, rwd_batch_n, nobs_batch_n, done_batch_n, self.categorical)
        loss_per_batch_elem += loss_per_batch_elem_n

        loss = torch.mean(loss_per_batch_elem * weights)
        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()
        return loss.item(), loss_per_batch_elem

    def calculate_loss_no_categorical(self, obs, act, rwd, nobs, done):
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
        loss = F.mse_loss(q_target, q, reduction='none')
        return loss

    def calculate_loss_categorical(self, obs, act, rwd, nobs, done):
        obs_batch = obs
        nobs_batch = nobs
        act_batch = act
        rwd_batch = rwd
        done_batch = done

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        with torch.no_grad():
            # Double DQN
            next_action = self.q_network(nobs_batch).argmax(1) if self.double_q else self.q_target_network(nobs_batch).argmax(1)
            next_dist = self.q_target_network.get_distribution(nobs_batch)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = rwd_batch + (1 - done_batch) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.q_network.get_distribution(obs_batch)
        log_p = torch.log(dist[range(self.batch_size), act_batch])
        loss_per_batch_elem = -(proj_dist * log_p).sum(1)
        return loss_per_batch_elem

    def train(self, batch, batch_n=None):
        obs_batch, act_batch, rwd_batch, nobs_batch, done_batch, weights, indexes = batch
        if self.use_n_step:
            obs_batch_n, act_batch_n, rwd_batch_n, nobs_batch_n, done_batch_n = batch_n
            act_batch_n, done_batch_n, nobs_batch_n, obs_batch_n, rwd_batch_n = self.transform_to_tensors(
                act_batch_n,
                done_batch_n,
                nobs_batch_n,
                obs_batch_n,
                rwd_batch_n)

        act_batch, done_batch, nobs_batch, obs_batch, rwd_batch = self.transform_to_tensors(act_batch, done_batch,
                                                                                            nobs_batch, obs_batch,
                                                                                            rwd_batch)

        if self.use_n_step:
            loss, td_error = self.update_q_with_n(obs_batch, act_batch, rwd_batch, nobs_batch, done_batch, weights, indexes
                                       ,obs_batch_n, act_batch_n, rwd_batch_n, nobs_batch_n, done_batch_n)
        else:
            loss, td_error = self.update_q(obs_batch, act_batch, rwd_batch, nobs_batch, done_batch, weights, indexes)
        self.update_target(self.q_target_network, self.q_network, p.tau)
        return loss, indexes, td_error

    def reset_net_noise(self):
        self.q_network.reset_noise()
        self.q_target_network.reset_noise()

    def transform_to_tensors(self, act_batch, done_batch, nobs_batch, obs_batch, rwd_batch):
        obs_batch = [np.array(obs) for obs in list(zip(*obs_batch))]
        obs_batch = [torch.FloatTensor(obs).to(self.device) for obs in obs_batch]
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rwd_batch = torch.FloatTensor(rwd_batch).to(self.device)
        nobs_batch = [np.array(obs) for obs in list(zip(*nobs_batch))]
        nobs_batch = [torch.FloatTensor(nobs).to(self.device) for nobs in nobs_batch]
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        return act_batch, done_batch, nobs_batch, obs_batch, rwd_batch

