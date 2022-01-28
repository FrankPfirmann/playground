import logging
from collections import deque

from agents.skynet_agents import SmartRandomAgent, SmartRandomAgentNoBomb
from logger import Logger
import random
from typing import Callable

import numpy as np
import pommerman
from pommerman.agents import SimpleAgent
from pommerman.constants import Item
import torch
import sys

import params as p
from util.data import transform_observation
from util.rewards import staying_alive_reward, go_down_right_reward, bomb_reward, skynet_reward, woods_close_to_bomb_reward
from agents.skynet_agents import SmartRandomAgent, SmartRandomAgentNoBomb
from agents.static_agent import StaticAgent
from agents.train_agent import TrainAgent
from agents.simple_agent_cautious_bomb import CautiousAgent
from data_augmentation import DataAugmentor
from logger import Logger
import params as p
from util.data import transform_observation
from util.rewards import staying_alive_reward, go_down_right_reward, bomb_reward, skynet_reward
from pommerman.constants import Action

class DataGeneratorPommerman:
    def __init__(self, env, augmentors: list=[])-> None:
        """
        Create a new DataGenerator instance.

        :param augmentors: A list of DataAugmentor derivates
        """
        self.device = torch.device("cpu")

        self.env = env
        # Define replay pool
        self.buffer = []
        self.episode_buffer = []
        self.episode_buffer_length = 0

        # prioritized replay variables
        if p.prioritized_replay:
            self.alpha = 1
            self.priority_sums = [[0 for _ in range(2 * p.replay_size)], [0 for _ in range(2 * p.replay_size)]] # store priorities in binary segment trees
            self.priority_mins = [[float('inf') for _ in range(2 * p.replay_size)], [float('inf') for _ in range(2 * p.replay_size)]] # store priorities in binary segment trees
            self.max_priorities = [1.0, 1.0]
            self.sizes = [0, 0]

        self.agents_n = 2 if self.env == 'OneVsOne-v0' or self.env.startswith("custom") else 4
        self.player_agents_n = int(self.agents_n/2)
        self.buffers = [[] for _ in range(self.player_agents_n)]
        self.idxs = [0, 0]
        self.augmentors = augmentors

        self.logger = Logger('log')

    def add_to_buffer(self, obs, act, rwd, nobs, done, agent_num):
        if len(self.buffers[agent_num]) < p.replay_size:
            self.buffers[agent_num].append([obs, act, [rwd], nobs, [done]])
        else:
            self.buffers[agent_num][self.idxs[agent_num]] = [obs, act, [rwd], nobs, [done]]

        if p.prioritized_replay:
            # set priority of new transitions to max_priority
            priority_alpha = self.max_priorities[agent_num] ** self.alpha

            self._set_priority_min(self.idxs[agent_num], priority_alpha, agent_num)
            self._set_priority_sum(self.idxs[agent_num], priority_alpha, agent_num)

        self.idxs[agent_num] = (self.idxs[agent_num] + 1) % p.replay_size

    def _set_priority_min(self, idx, priority_alpha, agent_num):
        ''' Update the minimum priotity tree'''
        # look at leaves of binary tree
        idx += p.replay_size 
        self.priority_mins[agent_num][idx] = priority_alpha
        # update whole tree
        while idx >= 2:
            idx //= 2 # index of parent 
            self.priority_mins[agent_num][idx] = min(self.priority_mins[agent_num][2 * idx], self.priority_mins[agent_num][2 * idx + 1])

    def _set_priority_sum(self, idx, priority_alpha, agent_num):
        ''' Update the maximum priority tree'''
        # look at leaves of tree
        idx += p.replay_size
        self.priority_sums[agent_num][idx] = priority_alpha
        # update whole tree
        while idx >= 2:
            idx //= 2 # index of parent
            self.priority_sums[agent_num][idx] = self.priority_sums[agent_num][2 * idx] + self.priority_sums[agent_num][2 * idx + 1]

    def _sum(self, agent_num):
        ''' get sum of priorities'''
        return self.priority_sums[agent_num][1]

    def _min(self, agent_num):
        ''' get min of priorities'''
        return self.priority_mins[agent_num][1]

    def find_prefix_sum_idx(self, prefix_sum, agent_num):
        ''' find smallest index, s.t. the sum up to that index is greater or equal to prefix_sum'''
        idx = 1 # start from root
        while idx < p.replay_size:
            if self.priority_sums[agent_num][idx * 2] > prefix_sum: # if sum of left branch is bigger, go to left branch
                idx = 2 * idx
            else:   # else go to right branch and substract sum of left branch
                prefix_sum -= self.priority_sums[agent_num][idx * 2] 
                idx = 2 * idx + 1
        return idx - p.replay_size

    def get_batch_buffer(self, size, agent_num, beta=p.beta):
        ''' sample transitions from buffer (including weights and indexes with prioritized replay '''
        if not p.prioritized_replay:
            batch = list(zip(*random.sample(self.buffers[agent_num], size)))
            return np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4])
        else:
            samples = {
                'weights': np.zeros(shape=size, dtype=np.float32),
                'indexes': np.zeros(shape=size, dtype=np.int32)
                }

            # sample indexes according to probability
            for i in range(size):
                prefix_sum = random.random() * self._sum(agent_num)
                idx = self.find_prefix_sum_idx(prefix_sum, agent_num)
                samples['indexes'][i] = idx

            # calculate max weight (used to calculate individual weights)
            prob_min = self._min(agent_num) / self._sum(agent_num)
            max_weight = (prob_min * self.sizes[agent_num]) ** (-beta)

            # calculate weights
            for i in range (size):
                idx = samples['indexes'][i]
                prob = self.priority_sums[agent_num][idx + p.replay_size] / self._sum(agent_num)
                weight = (prob * self.sizes[agent_num]) ** (-beta)
                samples['weights'][i] = weight / max_weight
            
            # get sample transitions
            transitions = list(zip(*np.array(self.buffers[agent_num])[samples['indexes']]))

            return np.array(transitions[0]), np.array(transitions[1]), np.array(transitions[2]), np.array(transitions[3]), np.array(transitions[4]), \
                samples['weights'], samples['indexes']

    def update_priorities(self, indexes, priorities, agent_num):
        ''' update priorities of transitions'''
        for idx, priority in zip(indexes, priorities):
            priority = abs(priority[0].item())
            self.max_priorities[agent_num] = max(self.max_priorities[agent_num], priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha, agent_num)
            self._set_priority_sum(idx, priority_alpha, agent_num)
        

    def get_batch_buffer_back(self, size, j):
        sample_pool = []
        for b in self.episode_buffer:
            sample_pool.extend(b[-j:])
        batch = list(zip(*random.sample(sample_pool, size)))
        return np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4])

    def add_to_episode_buffer(self, i, obs, act, rwd, nobs, done):
        if len(self.buffer) < p.replay_size:
            self.buffers[i].append([obs, act, [rwd], nobs, [done]])
        else:
            self.buffers[i][self.idx] = [obs, act, [rwd], nobs, [done]]
        self.idx = (self.idx + 1) % p.replay_size

    def get_episode_buffer(self):
        batch = list(zip(*random.sample(self.episode_buffer, 1)[0]))
        return np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4])

    def _init_agent_list(self, agent1, agent2, policy1, policy2, enemy, setposition=False):
        '''
        Helper method for creating agent_list
        :param agent1: string identifying agent1
        :param agent2: string identifying agent2
        :param policy1: policy the first train agent follows
        :param policy2: policy the second train agent follows
        :param enemy: string identifying the enemy
        :param setPosition: whether we initialize agent1 always on top left or randomly
        :return: agent indexes, igent ids on board observation and agent list of agent objects
        '''
        agent_list = [None] * self.agents_n
        agent_ind = 0 if setposition else np.random.randint(2)
        for i in range(0, self.agents_n):

            if i == agent_ind:
                agent_str = agent1
            elif i == agent_ind + 2:
                agent_str = agent2
            else:
                agent_str = enemy

            if agent_str.startswith('static'):
                _, action = agent_str.split(':')
                agent_list[i] = StaticAgent(int(action))
            elif agent_str == 'train' and i == agent_ind:
                agent_list[i] = TrainAgent(policy1)
            elif agent_str == 'train' and i == agent_ind + 2:
                agent_list[i] = TrainAgent(policy2)
            elif agent_str == 'smart_random':
                agent_list[i] = SmartRandomAgent()
            elif agent_str == 'smart_random_no_bomb':
                agent_list[i] = SmartRandomAgentNoBomb()
            elif agent_str == 'simple':
                agent_list[i] = SimpleAgent()
            elif agent_str == 'cautious':
                agent_list[i] = CautiousAgent()
            else:
                logging.error('unsupported opponent type!')
                sys.exit(1)
        if self.agents_n == 2:
            agent_inds = [0 + agent_ind]
            agent_ids = [10 + agent_ind]
        else:
            agent_inds = [0 + agent_ind, 2 + agent_ind]
            agent_ids = [10 + agent_ind, 12 + agent_ind]
        return agent_inds, agent_ids, agent_list

    def generate(self, episodes: int, policy1: Callable, policy2: Callable, enemy: str, transformer: Callable, agent1, agent2, max_steps: int=p.max_steps, render: bool=False) -> tuple:
        """
        Generate ``episodes`` samples acting by ``policy`` and saving
        observations transformed with ``transformer``.

        :param episodes: The number of episodes to generate
        :param policy: A callable policy to use for action selection
        :param: String identifying the enemy
        :param transformer: A callable transformer to use for input
            transformation
        :param render: If ``True`` the environment will be rendered in a
            window, otherwise not.

        :return: A tuple containing the results of generation in the
            following order: list of wins, ties, average reward, action
            counts, average steps
        """

        res = np.array([0.0] * 2)
        act_counts = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ties = 0.0
        avg_rwd = 0.0
        avg_steps = 0.0
        fifo = [[] for _ in range(self.agents_n)]
        skynet_reward_log = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        for i_episode in range(episodes):
            agent_inds, agent_ids, agent_list = self._init_agent_list(agent1, agent2, policy1, policy2, enemy, True)

            env = pommerman.make(self.env, agent_list)
            obs = env.reset()
            done = False
            ep_rwd = 0.0
            alive = [True, True]
            steps_n = 0
            for i in range(self.agents_n):
                fifo[i].clear()

            while not done:
                if render and i_episode == 0:
                    env.render()
                act = env.act(obs)
                for j in range(self.player_agents_n):
                    act_counts[j][int(act[agent_inds[j]])] += 1
                nobs, rwd, done, _ = env.step(act)
                if p.reward_func == "SkynetReward":
                    skynet_rwds = skynet_reward(obs, act, nobs, fifo, agent_inds, skynet_reward_log)

                for i in range(self.player_agents_n):
                    if p.reward_func == "SkynetReward":
                        agt_rwd = skynet_rwds[agent_inds[i]]
                    elif p.reward_func == "BombReward":
                        agt_rwd = bomb_reward(nobs, act, agent_inds[i])/100
                    else:
                        agt_rwd = staying_alive_reward(nobs, agent_ids[i])
                    #only living agent gets winning rewards
                    if done:
                        winner = np.where(np.array(rwd) == 1)[0] # TODO even dead agents get reward?
                        if agent_inds[0] in winner:
                            agt_rwd = 0.5
                            logging.info(f"Win rewarded with {agt_rwd} for each living agent")
                    #draw reward for living agents
                    if steps_n == max_steps:
                        done = True
                        if agent_list[agent_inds[i]].is_alive:
                            agt_rwd = 0.0
                            logging.info(f"Draw rewarded with {agt_rwd} for each living agent")
                    #death reward
                    if alive[i] and agent_ids[i] not in nobs[agent_inds[i]]['alive']:
                        agt_rwd = -0.5
                        logging.info(f"Death of agent {agent_inds[i]} rewarded with {agt_rwd}")
                    if alive[i]:
                        # Build original transition
                        transition = (transformer(obs[agent_inds[i]]), act[agent_inds[i]], agt_rwd*100, \
                                      transformer(nobs[agent_inds[i]]), done)
                        transitions = [transition]
                        # Create new transitions
                        for augmentor in self.augmentors:
                            transition_augmented = augmentor.augment(obs[agent_inds[i]], act[agent_inds[i]], agt_rwd*100, nobs[agent_inds[i]], not alive)
                            for t in transition_augmented:
                                transitions.append((transformer(t[0]), t[1], t[2]*100, transformer(t[3]), t[4]))

                        # Add everything to the buffer
                        for t in transitions:
                            if p.backplay:
                                self.add_to_episode_buffer(i, *t)
                            if not p.episode_backward:
                                self.add_to_buffer(*t, i)
                            else:
                                self.add_to_episode_buffer(i, *t)

                    if alive[i]:
                        ep_rwd += agt_rwd
                    alive[i] = agent_ids[i] in nobs[agent_inds[i]]['alive']
                obs = nobs
                steps_n += 1
            #add to episode buffer
            if p.episode_backward or p.backplay:
                for i in range(len(self.buffers)):
                    self.episode_buffer.append(self.buffers[i])
                    if self.episode_buffer_length + len(self.buffers[i]) > p.replay_size:
                        self.episode_buffer.pop(0)
                    self.episode_buffer_length += len(self.buffers[i])
                    self.buffers[i] = []
            avg_rwd += ep_rwd
            avg_steps += steps_n
            winner = np.where(np.array(rwd) == 1)[0]
            if len(winner) == 0:
                ties += 1
            else:
                k = True
                for i in range(self.player_agents_n):
                    if agent_inds[i] in winner:
                        res[0] += 1
                        k = False
                        break
                if k:
                    res[1] += 1

            env.close()
        avg_rwd /= episodes
        avg_steps /= episodes
        logging.info(f"Wins: {res}, Ties: {ties}, Avg. Reward: {avg_rwd}, Avg. Steps: {avg_steps}")
        if p.reward_func == "SkynetReward":
            logging.info(
                f"Skynet Reward split agent 1: Kills: {skynet_reward_log[0][0]}, Items: {skynet_reward_log[0][1]},"
                        f" Steps(FIFO): {skynet_reward_log[0][2]}, Bombs: {skynet_reward_log[0][3]}, Deaths: {skynet_reward_log[0][4]}")
            logging.info(
                f"Skynet Reward split agent 2: Kills: {skynet_reward_log[1][0]}, Items: {skynet_reward_log[1][1]},"
                        f" Steps(FIFO): {skynet_reward_log[1][2]}, Bombs: {skynet_reward_log[1][3]}, Deaths: {skynet_reward_log[1][4]}")
        logging.info(act_counts)
        self.logger.write(res, ties, avg_rwd)
        # TODO: Change the return type to something more readable outside the function
        return (res, ties, avg_rwd, act_counts, avg_steps)