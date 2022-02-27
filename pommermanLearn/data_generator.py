import logging

from typing import Callable

import numpy as np
import pommerman
from pommerman.agents import SimpleAgent
import torch
import sys

from agents.skynet_agents import SmartRandomAgent, SmartRandomAgentNoBomb
from agents.static_agent import StaticAgent
from agents.train_agent import TrainAgent
from agents.simple_agent_cautious_bomb import CautiousAgent
from logger import Logger
import params as p
from util.rewards import staying_alive_reward, bomb_reward, skynet_reward
from util.replay_buffer import ReplayBuffer
from util.analytics import Stopwatch


class DataGeneratorPommerman:
    def __init__(self, env, augmentors: list=[])-> None:
        """
        Create a new DataGenerator instance.

        :param augmentors: A list of DataAugmentor derivates
        """
        self.device = torch.device("cpu")

        self.env = env
        # Define replay pool
        self.replay_buffers = [ReplayBuffer(p.replay_size), ReplayBuffer(p.replay_size)]

        self.agents_n = 2 if self.env == 'OneVsOne-v0' else 4
        self.player_agents_n = int(self.agents_n/2)
        self.augmentors = augmentors

        self.logger = Logger('log')

    def get_batch_buffer(self, batch_size, agent_num):
        return self.replay_buffers[agent_num].get_batch_buffer(batch_size)

    def update_priorities(self, indexes, td_error, agent_num):
        return self.replay_buffers[agent_num].update_priorities(indexes, td_error)

    def _init_agent_list(self, agent1, agent2, policy1, policy2, enemy, transformer, setposition=False):
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
                agent_list[i] = TrainAgent(policy1, transformer)
            elif agent_str == 'train' and i == agent_ind + 2:
                agent_list[i] = TrainAgent(policy2, transformer)
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
        for _ in range(episodes):
            reward, steps, res, ties, act_counts, skynet_reward_log = self.generate_episode(agent1, agent2, policy1, policy2, enemy, transformer, max_steps)
            total_reward += reward
            total_steps += steps
        average_reward = total_reward/episodes
        average_steps = total_steps/episodes

        #print(f"{stopwatch.stop()}s for sequential execution")

        logging.info(f"Wins: {res}, Ties: {ties}, Avg. Reward: {average_reward}, Avg. Steps: {average_steps}")
        if p.reward_func == "SkynetReward":
            logging.info(
                f"Skynet Reward split agent 1: Kills: {skynet_reward_log[0][0]}, Items: {skynet_reward_log[0][1]},"
                        f" Steps(FIFO): {skynet_reward_log[0][2]}, Bombs: {skynet_reward_log[0][3]}, Deaths: {skynet_reward_log[0][4]}")
            logging.info(
                f"Skynet Reward split agent 2: Kills: {skynet_reward_log[1][0]}, Items: {skynet_reward_log[1][1]},"
                        f" Steps(FIFO): {skynet_reward_log[1][2]}, Bombs: {skynet_reward_log[1][3]}, Deaths: {skynet_reward_log[1][4]}")
        logging.info(act_counts)
        self.logger.write(res, ties, average_reward)
        # TODO: Change the return type to something more readable outside the function
        return (res, ties, average_reward, act_counts, average_steps)

    def generate_episode(self, agent1, agent2, policy1, policy2, enemy, transformer, max_steps):
        res = np.array([0.0] * 2)
        act_counts = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ties = 0.0
        reward = 0.0
        steps = 0.0

        agent_inds, agent_ids, agent_list = self._init_agent_list(agent1, agent2, policy1, policy2
                                                                    , enemy, transformer, p.set_position)
        env = pommerman.make(self.env, agent_list)
        obs = env.reset()
        done = False
        ep_rwd = 0.0
        alive = [True, True]
        steps_n = 0

        # Needed for skynet rewards
        self.reward_log = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        self.fifo = [[] for _ in range(self.agents_n)]

        while not done:
            act = env.act(obs)
            first_act = [i[0] if type(i) == list else i for i in act]
            for j in range(self.player_agents_n):
                act_counts[j][int(first_act[agent_inds[j]])] += 1
            nobs, rwd, done, _ = env.step(act)
            
            rewards = self.calculate_rewards(obs, act, nobs, rwd, done)

            for i in range(self.player_agents_n):
                agt: TrainAgent = agent_list[agent_inds[i]]
                agt_rwd = rewards[agent_inds[i]]
                if alive[i]:
                    # Build original transition
                    if p.use_memory:
                        agt_obs = agt.get_memory_view()
                        agt.update_memory(nobs[agent_inds[i]])
                        agt_nobs = agt.get_memory_view()
                        act_no_msg = act[agent_inds[i]][0] if p.communicate else act[agent_inds[i]]
                        transition = (agt_obs, act_no_msg, agt_rwd * 100, \
                                        agt_nobs, done)
                    else:
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
                        self.replay_buffers[i].add_to_buffer(*t)

                if alive[i]:
                    ep_rwd += agt_rwd
                alive[i] = agent_ids[i] in nobs[agent_inds[i]]['alive']
            obs = nobs
            steps_n += 1
        reward += ep_rwd
        steps += steps_n
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
        return (reward, steps, res, ties, act_counts, self.reward_log)
    
    def calculate_rewards(self, obs: dict, act: list, nobs: dict, env_reward: float, done: bool):
        rewards = []
        indices = [i for i in range(self.agents_n)]
        ids     = [10+i for i in indices]

        for index, id in zip(indices, ids):
            alive   = id in obs[index]['alive']
            died    = alive and id not in obs[index]['alive']
            steps_n = obs[index]['step_count']
            won     = env_reward[index] == 1
            agt_rwd = 0.0

            if p.reward_func == "SkynetReward":
                agt_rwd = skynet_reward(obs, act, nobs, self.fifo, [index], self.reward_log)[index]
            elif p.reward_func == "BombReward":
                agt_rwd = bomb_reward(nobs, act, index)/100
            else:
                agt_rwd = staying_alive_reward(nobs, id)

            # woods close to bomb reward
            # if act[agent_inds[i]] == Action.Bomb.value:
            #     agent_obs = obs[agent_inds[i]]
            #     agt_rwd += woods_close_to_bomb_reward(agent_obs, agent_obs['position'], agent_obs['blast_strength'], agent_ids)
            #only living agent gets winning rewards
            if done:
                if won:
                    agt_rwd = 0.5
                    logging.info(f"Win rewarded with {agt_rwd} for each living agent")

            #draw reward for living agents
            if steps_n == p.max_steps:
                done = True
                if alive:
                    agt_rwd = 0.0
                    logging.info(f"Draw rewarded with {agt_rwd} for each living agent")

            # Negative reward if agent died this iteration
            if died:
                agt_rwd = -0.5
                logging.info(f"Death of agent {index} rewarded with {agt_rwd}")

            rewards.append(agt_rwd)
        return rewards