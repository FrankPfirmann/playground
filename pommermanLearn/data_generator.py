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
from util.analytics import Stopwatch
from util.bomb_tracker import BombTracker
from util.rewards import staying_alive_reward, bomb_reward, skynet_reward, woods_close_to_bomb_reward
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
        if p.use_nstep:
            self.replay_buffers_n = [ReplayBuffer(p.replay_size, p.nsteps), ReplayBuffer(p.replay_size, p.nsteps)]
        self.agents_n = 2 if self.env == 'OneVsOne-v0' else 4
        self.player_agents_n = int(self.agents_n/2)
        self.augmentors = augmentors

        self.logger = Logger('log')
        self.init_session()


    def init_session(self):
        self.reward_log = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        self.act_counts = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.res = np.array([0.0] * 2)
        self.ties = 0.0

    def get_batch_buffer(self, batch_size, agent_num):
        return self.replay_buffers[agent_num].get_batch_buffer(batch_size)

    def get_batch_buffer_from_idx(self, idxs, agent_num):
        return self.replay_buffers_n[agent_num].get_batch_from_indices(idxs)

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

    def generate(self, episodes: int, policy1: Callable, policy2: Callable, enemy: str, transformer: Callable,
                 agent1, agent2, max_steps: int=p.max_steps, render: bool=False, test: bool=False) -> tuple:
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
        total_reward = 0.0
        total_steps  = 0.0

        self.init_session()
        for _ in range(episodes):
            reward, steps = \
                self.generate_episode(agent1, agent2, policy1, policy2, enemy, transformer, max_steps,
                                      test=test, render=render)
            total_reward += reward
            total_steps += steps
        average_reward = total_reward/episodes
        average_steps = total_steps/episodes

        logging.info(f"Wins: {self.res}, Ties: {self.ties}, Avg. Reward: {average_reward}, Avg. Steps: {average_steps}")
        if p.reward_func == "SkynetReward":
            logging.info(
                f"Skynet Reward split agent 1: Win/loss: {self.reward_log[0][3]}, Kills: {self.reward_log[0][0]}, Items: {self.reward_log[0][1]},"
                        f" Steps(FIFO): {self.reward_log[0][2]}, Deaths: {self.reward_log[0][4]}")
            logging.info(
                f"Skynet Reward split agent 2: Win/loss: {self.reward_log[1][3]}, Kills: {self.reward_log[1][0]}, Items: {self.reward_log[1][1]},"
                        f" Steps(FIFO): {self.reward_log[1][2]}, Deaths: {self.reward_log[1][4]}")
        logging.info(self.act_counts)
        self.logger.write(self.res, self.ties, average_reward)
        # TODO: Change the return type to something more readable outside the function
        return self.res, self.ties, average_reward, self.act_counts, average_steps

    def generate_episode(self, agent1: str, agent2: str, policy1, policy2, enemy: str, transformer, max_steps,
                         test=False, render=False):
        """
        Play an episode in the Pommerman environment and returns the accumulated results.

        :param agent1: String denoting the first player agent
        :param agent2: String denoting the second player agent
        :param policy1: `Callable` to get actions from for `agent1`
        :param policy2: `Callable` to get actions from for `agent2`
        :param enemy: String denoting both enemy agents
        :param transformer: `Callable` to transform the environment
            observation to a suitable format for the policy.
        """
        reward = 0.0
        steps = 0.0

        agent_inds, agent_ids, agent_list = self._init_agent_list(agent1, agent2, policy1, policy2
                                                                    , enemy, transformer, p.set_position)
        env = pommerman.make(self.env, agent_list)
        obs = env.reset()
        done = False
        ep_rwd = 0.0
        was_alive = [True, True]
        steps_n = 0

        bomb_tracker = BombTracker()
        # Needed for skynet rewards
        self.fifo = [[] for _ in range(self.agents_n)]

        while not done:
            if render:
                env.render()
            act = env.act(obs)
            first_act = [i[0] if type(i) == list else i for i in act]
            for j in range(self.player_agents_n):
                self.act_counts[j][int(first_act[agent_inds[j]])] += 1
            nobs, rwd, done, _ = env.step(act)
            bomb_tracker.step(obs, nobs)
            if p.reward_func == "SkynetReward":
                rewards = skynet_reward(obs, act, nobs, self.fifo, agent_inds, self.reward_log, done, bomb_tracker)

            for i in range(self.player_agents_n):
                    agt_id  = agent_ids[i]
                    agt_idx = agent_inds[i]
                    agt     = agent_list[agt_idx]
                    agt_rwd = rewards[agt_idx]

                    if was_alive[i]:
                        if steps_n == max_steps:
                            done = True
                            if agt.is_alive:
                                logging.info(f"Draw")
                        # Build original transition
                        if p.use_memory:
                            agt_obs = agt.get_memory_view()
                            agt.update_memory(nobs[agt_idx])
                            agt_nobs = agt.get_memory_view()
                            act_no_msg = act[agt_idx][0] if p.communicate else act[agt_idx]
                            transition = (agt_obs, act_no_msg, agt_rwd, \
                                            agt_nobs, done)
                        else:
                            transition = (transformer(obs[agt_idx]), act[agt_idx], agt_rwd, \
                                            transformer(nobs[agt_idx]), done)
                        transitions = [transition]

                        # Apply data augmentation
                        transitions.extend(self.augment_transition((obs[agt_idx], act[agt_idx], agt_rwd, nobs[agt_idx], not was_alive), transformer))

                        # Add everything to the buffer
                        if not test:
                            for t in transitions:
                                if p.use_nstep:
                                    # only add to 1-step-buffer
                                    added_to_n = self.replay_buffers_n[i].add_to_buffer(*t)
                                    if added_to_n:
                                        self.replay_buffers[i].add_to_buffer(*t)
                                else:
                                    self.replay_buffers[i].add_to_buffer(*t)

                    if was_alive[i]:
                        ep_rwd += agt_rwd
                    was_alive[i] = agt_id in nobs[agt_idx]['alive']
            obs = nobs
            steps_n += 1
        reward += ep_rwd
        steps += steps_n
        winner = np.where(np.array(rwd) == 1)[0]
        if len(winner) == 0:
            self.ties += 1
        else:
            k = True
            for i in range(self.player_agents_n):
                if agt_idx in winner:
                    self.res[0] += 1
                    k = False
                    break
            if k:
                self.res[1] += 1

            env.close()

        return reward, steps


    def augment_transition(self, transition: list, transformer: Callable) -> list:
        """
        Apply augmentors to the given transition and transform them.

        :param self:
        :param transition: Transition to augment
        :param transformer: Transformer to apply afterwards
        """
        transitions = []
        for augmentor in self.augmentors:
            augmented = augmentor.augment(*transition)
            for a in augmented:
                transitions.append((transformer(a[0]), a[1], a[2] * 100, transformer(a[3]), a[4]))
        return transitions


    def calculate_rewards(self, obs: dict, act: list, nobs: dict, env_reward: float, done: bool):
        """
        Calculate the rewards for the given state transitions.

        :param obs: `List` containing observations of each agent
        :param act: List containing the chosen actions by all agents
        :param nobs: `List` containing the observation following `obs`
            for each agent.
        :param env_reward: Reward given directly from the environment.
        :param done: `True` if the episode terminated, otherwise `False`
        """
        rewards = []
        indices = [i for i in range(self.agents_n)]
        ids = [10 + i for i in indices]
        for index, id in zip(indices, ids):
            alive = id in obs[index]['alive']
            died = alive and id not in obs[index]['alive']
            steps_n = obs[index]['step_count']
            won = env_reward[index] == 1
            agt_rwd = 0.0

            if p.reward_func == "SkynetReward":
                agt_rwd = skynet_reward(obs, act, nobs, self.fifo, [index], self.reward_log)[index]
            elif p.reward_func == "BombReward":
                agt_rwd = bomb_reward(nobs, act, index) / 100
            else:
                agt_rwd = staying_alive_reward(nobs, id)

            # woods close to bomb reward
            # if act[agent_inds[i]] == Action.Bomb.value:
            #     agent_obs = obs[agent_inds[i]]
            #     agt_rwd += woods_close_to_bomb_reward(agent_obs, agent_obs['position'], agent_obs['blast_strength'], agent_ids)
            # only living agent gets winning rewards
            if done:
                if won:
                    agt_rwd = 0.5
                    logging.info(f"Win rewarded with {agt_rwd} for each living agent")

            # draw reward for living agents
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