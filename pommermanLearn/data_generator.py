import logging
from logger import Logger
import random
from typing import Callable

import numpy as np
import pommerman
from pommerman.constants import Item
import torch

import params as p
from util.data import transform_observation
from util.rewards import staying_alive_reward,go_down_right_reward, bomb_reward
from agents.static_agent import StaticAgent
from agents.train_agent import TrainAgent

class DataGeneratorPommerman:
    def __init__(self, env):
        self.device = torch.device("cpu")

        # Define replay pool
        self.buffer = []
        self.episode_buffer = []
        self.idx = 0
        self.env = env

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
        # Assume first agent is TrainAgent
        agent_list = [
            TrainAgent(policy),
            StaticAgent(0)
        ]
        agent_id=10 # ID of training agent
        env = pommerman.make(self.env, agent_list)

        res = np.array([0.0, 0.0])
        act_counts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ties = 0.0
        avg_rwd = 0.0
        avg_steps=0.0
        for i_episode in range(episodes):
            obs = env.reset()
            done = False
            ep_rwd = 0.0
            dead_before = False
            steps_n = 0
            # high pos for down right reward
            high_pos=(0,0)
            while not done and steps_n < p.max_steps:
                if render and i_episode == 0:
                    env.render()
                act = env.act(obs)
                act_counts[int(act[0])] += 1
                nobs, rwd, done, _ = env.step(act)
                if p.reward_func == "DownRight":
                    agt_rwd, high_pos = go_down_right_reward(nobs, high_pos, 0, act)
                elif p.reward_func == "BombReward":
                    agt_rwd = bomb_reward(nobs, act)
                else:
                    agt_rwd = staying_alive_reward(nobs, agent_id)
                if agent_id in nobs[0]['alive']: #if train agent is still alive
                    self.add_to_buffer(transformer(obs[0]), act[0], agt_rwd, transformer(nobs[0]), False)
                    ep_rwd += agt_rwd
                else:
                    if not dead_before:
                        self.add_to_buffer(transformer(obs[0]), act[0], agt_rwd, transformer(nobs[0]), True)
                    dead_before = True

                #self.add_to_buffer(obs[0], act[0], rwd[0], nobs[0], done)
                obs = nobs
                steps_n += 1
            if p.episode_backward:
                self.episode_buffer.append(self.buffer)
                self.buffer = []
            avg_rwd += ep_rwd
            avg_steps+=steps_n
            winner = np.where(np.array(rwd) == 1)[0]
            if len(winner) == 0:
                ties += 1
            else:
                res[winner] += 1

        avg_rwd /= episodes
        avg_steps /= episodes
        logging.info(f"Wins: {res}, Ties: {ties}, Avg. Reward: {avg_rwd}, Avg. Steps: {avg_steps}")
        logging.info(act_counts)
        self.logger.write(res, ties, avg_rwd)
        env.close()
        # TODO: Change the return type to something more readable outside the function
        return (res, ties, avg_rwd, act_counts, avg_steps)
