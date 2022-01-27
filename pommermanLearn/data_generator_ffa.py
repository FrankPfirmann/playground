from argparse import Action
import logging
from collections import deque

from logger import Logger
import random
from typing import Callable

import numpy as np
import pommerman
from pommerman.constants import Item, Action
import torch
import sys

import params as p
from util.data import transform_observation
from util.rewards import staying_alive_reward, go_down_right_reward, bomb_reward, skynet_reward
from agents.skynet_agents import SmartRandomAgent
from agents.train_agent import TrainAgent
from agents.static_agent import StaticAgent
from pommerman.agents import SimpleAgent
from agents.simple_agent_cautious_bomb import CautiousAgent
from data_augmentation import DataAugmentor

class DataGeneratorPommermanFFA:
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
            agent_list = [
                        TrainAgent(policy),
                        StaticAgent(0),
                        StaticAgent(0),
                        StaticAgent(0),
                        ]

            env = pommerman.make('PommeFFACompetition-v0', agent_list)
            obs = env.reset()
            done = False
            ep_rwd = 0.0
            alive = True
            steps_n = 0
            for i in range(self.agents_n):
                fifo[i].clear()

            while not done:
                if render and i_episode == 0:
                    env.render()
                act = env.act(obs)
                act_counts[int(act[0])] += 1
                nobs, rwd, done, _ = env.step(act)

                # abort game after max_steps
                if steps_n == p.max_steps:
                    done = True

                if p.reward_func == "SkynetReward":
                    skynet_rwds = skynet_reward(obs, act, nobs, fifo, [0], skynet_reward_log)
                    agt_rwd = skynet_rwds[0]

                if alive:
                    agt_rwd += rwd[0]
                    # Build original transition
                    transition = (transformer(obs[0]), act[0], agt_rwd*100, transformer(nobs[0]), done)
                    transitions = [transition]
                    # Create new transitions
                    for augmentor in self.augmenter:
                        transition_augmented = augmentor.augment(obs[0], act[0], agt_rwd*100, nobs[0], done)
                        for t in transition_augmented:
                            transitions.append((transformer(t[0]), t[1], t[2]*100, transformer(t[3]), t[4]))

                    # Add everything to the buffer
                    # for _ in range(int(steps_n/10)): # later steps are added more often
                    for t in transitions:
                        if not p.episode_backward:
                            self.add_to_buffer(*t)
                        else:
                            self.add_to_episode_buffer(i, *t)

                    alive = 10 in nobs[0]['alive']
                    if alive:
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
            elif 0 in winner:
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
