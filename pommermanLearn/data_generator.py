import logging
from collections import deque

from logger import Logger
import random
from typing import Callable

import numpy as np
import pommerman
from pommerman.constants import Item
import torch
import sys

import params as p
from util.data import transform_observation
from util.rewards import staying_alive_reward, go_down_right_reward, bomb_reward, skynet_reward
from agents.static_agent import StaticAgent
from agents.train_agent import TrainAgent
from agents.simple_agent_cautious_bomb import CautiousAgent
from data_augmentation import DataAugmentor

class DataGeneratorPommerman:
    def __init__(self, env, augmenter: list=[])-> None:
        """
        Create a new DataGenerator instance.

        :param augmenter: A list of DataAugmentor derivates
        """
        self.device = torch.device("cpu")


        self.env = env
        # Define replay pool
        self.buffer = []
        self.episode_buffer = []

        self.agents_n = 2 if self.env == 'OneVsOne-v0' else 4
        self.buffers = [[] for _ in range(int(self.agents_n/2))]
        self.idx = 0
        self.augmenter = augmenter

        self.logger = Logger('log')

    def add_to_buffer(self, obs, act, rwd, nobs, done):
        if len(self.buffer) < p.replay_size:
            self.buffer.append([obs, act, [rwd], nobs, [done]])
        else:
            self.buffer[self.idx] = [obs, act, [rwd], nobs, [done]]
        self.idx = (self.idx + 1) % p.replay_size

    def get_batch_buffer(self, size):
        batch = list(zip(*random.sample(self.buffer, size)))
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


    def generate(self, episodes: int, policy: Callable, transformer: Callable, render: bool=False) -> tuple:
        """
        Generate ``episodes`` samples acting by ``policy`` and saving
        observations transformed with ``transformer``.

        :param episodes: The number of episodes to generate
        :param policy: A callable policy to use for action selection
        :param transformer: A callable transformer to use for input
            transformation
        :param render: If ``True`` the environment will be rendered in a
            window, otherwise not.

        :return: A tuple containing the results of generation in the
            following order: list of wins, ties, average reward, action
            counts, average steps
        """


        res = np.array([0.0] * 2)
        act_counts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ties = 0.0
        avg_rwd = 0.0
        avg_steps = 0.0
        fifo = [[] for _ in range(self.agents_n)]
        skynet_reward_log = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        for i_episode in range(episodes):
            agent_ind = np.random.randint(0, 1)
            agent_list1 = [
                        TrainAgent(policy),
                        CautiousAgent(),
                        TrainAgent(policy),
                        CautiousAgent(),
                        ]

            agent_list2 = [
                        CautiousAgent(),
                        TrainAgent(policy),
                        CautiousAgent(),
                        TrainAgent(policy)
                        ]
            # get indices of agents
            agent_inds = [0+agent_ind, 2+agent_ind]
            agent_ids = [10+agent_ind, 12+agent_ind]

            agent_list = agent_list1 if agent_ind == 0 else agent_list2
            agent_id = 10 + agent_ind  # ID of training agent

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
                act_counts[int(act[agent_inds[0]])] += 1
                act_counts[int(act[agent_inds[1]])] += 1
                nobs, rwd, done, _ = env.step(act)
                if p.reward_func == "SkynetReward":
                    skynet_rwds = skynet_reward(obs, act, nobs, fifo, agent_inds, skynet_reward_log)

                for i in range(2):
                    if p.reward_func == "SkynetReward":
                        agt_rwd = skynet_rwds[agent_inds[i]]
                    elif p.reward_func == "BombReward":
                        agt_rwd = bomb_reward(nobs, act, agent_inds[i])/100
                    else:
                        agt_rwd = staying_alive_reward(nobs, agent_ids[i])

                    #draw reward
                    if done and steps_n == pommerman.constants.MAX_STEPS:
                        if agent_list[agent_inds[i]].is_alive:
                            agt_rwd = -1
                            logging.info(f"Draw rewarded with {agt_rwd} for each living agent")
                    #death reward
                    if alive[i] and agent_ids[i] not in nobs[agent_inds[i]]['alive']:
                        agt_rwd = -1
                        logging.info(f"Death of agent {agent_inds[i]} rewarded with {agt_rwd}")

                    if alive[i]:
                        # Build original transition
                        transition = (transformer(obs[agent_inds[i]]), act[agent_inds[i]], agt_rwd*100, transformer(nobs[agent_inds[i]]), not alive)
                        transitions = [transition]
                        # Create new transitions
                        for augmentor in self.augmenter:
                            transitions.extend( augmentor.augment(*transition) )

                        # Add everything to the buffer
                        for t in transitions:
                            if not p.episode_backward:
                                self.add_to_buffer(*t)
                            else:
                                self.add_to_episode_buffer(i, *t)

                    alive[i] = agent_ids[i] in nobs[agent_inds[i]]['alive']
                    if alive[i]:
                        ep_rwd += agt_rwd

                obs = nobs
                steps_n += 1
            if p.episode_backward:
                for i in range(len(self.buffers)):
                    self.episode_buffer.append(self.buffers[i])
                    self.buffers[i] = []
            avg_rwd += ep_rwd
            avg_steps += steps_n
            winner = np.where(np.array(rwd) == 1)[0]
            if len(winner) == 0:
                ties += 1
            elif agent_inds[0] in winner or agent_inds[1] in winner:
                res[0] += 1
            else:
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
