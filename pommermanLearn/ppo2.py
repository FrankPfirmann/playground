import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from action_prune import get_filtered_actions
from models import ActorCritic
import params as p
from util.data import transform_observation_simple, transform_observation_partial, transform_observation_centralized



class PPO(object):

    def __init__(self, network: ActorCritic, init_exploration,
                 is_train: bool=True, device: torch.device=None):
        """
        Create a new PPO model for training or inference.

        :param network: The actor critic train network
        :param is_train: If True, random exploration is enabled
        :param device: If set, use the given device. Otherwise use the CPU.
        """

        if p.p_observable:
            self.transform_func = transform_observation_partial
        elif p.centralize_planes:
            self.transform_func = transform_observation_centralized
        else:
            self.transform_func = transform_observation_simple

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # Define q and q_target
        self.policy = network
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': p.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': p.lr_critic}
                    ])   

        self.policy_old = ActorCritic(p.p_observable or p.centralize_planes, self.transform_func)
        self.policy_old.load_state_dict(self.policy.state_dict())     


        self.is_train = is_train
        self.exploration = init_exploration

        self.MseLoss = nn.MSELoss()

    def get_policy(self):
        def policy(obs):
            # get valid actions according to action filter and transform to be able to filter tensor
            valid_actions = get_filtered_actions(obs)
            valid_actions_transformed = []
            for i in range(6):
                valid_actions_transformed += [1] if i in valid_actions else [0]
            valid_actions_transformed = torch.FloatTensor(valid_actions_transformed).to(self.device).unsqueeze(0)

            obs = transform_observation_partial(obs)
            
            with torch.no_grad():
                obs = torch.FloatTensor(obs).to(self.device) #.unsqueeze(0).unsqueeze(0)
                action, action_logprob = self.policy_old(obs, valid_actions_transformed)

            return action, action_logprob, obs
        return policy

    def act(self, state):

        state = self.transform_func(state)

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        

        return state, action, action_logprob

    def set_train(self, t):
        self.is_train = t

    def set_exploration(self, exp):
        pass

    def update(self, buffers):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffers["rewards"]), reversed(buffers["dones"])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (p.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)


        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffers["states"], dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(buffers["actions"], dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(buffers["logprobs"], dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(p.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-p.eps_clip, 1+p.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss.mean()

    def _init_target_table(self, nobs_batch):
        with torch.no_grad():
            nobs_batch = [torch.FloatTensor(nobs).to(self.device) for nobs in list(zip(*nobs_batch))]
            q_t = self.q_target_network(nobs_batch).squeeze().detach().numpy()
            return q_t

    def train(self, batch):

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
        loss, td_error = self.update(obs_batch, act_batch, rwd_batch, nobs_batch, done_batch, weights)
        
        return loss, indexes, td_error

