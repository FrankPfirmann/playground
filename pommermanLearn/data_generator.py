import random
import numpy as np
import torch
import pommerman

from pommerman import agents
from logger import Logger


replay_size = 100000
exploration_noise = 0.5


class DataGeneratorGymDiscrete:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cpu")

        self.obs_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.n

        # Define replay pool
        self.buffer = []
        self.idx = 0

    def add_to_buffer(self, obs, act, rwd, nobs, done):
        if len(self.buffer) < replay_size:
            self.buffer.append([obs, act, [rwd], nobs, [done]])
        else:
            self.buffer[self.idx] = [obs, act, [rwd], nobs, [done]]
        self.idx = (self.idx + 1) % replay_size

    def get_batch_buffer(self, size):
        batch = list(zip(*random.sample(self.buffer, size)))
        return np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4])

    def get_action(self, policy, obs):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        act = policy(obs)
        return act.detach().numpy()[0]

    def generate(self, episodes, policy):
        avg_rwd = 0.0
        for i in range(episodes):
            done = False
            obs = self.env.reset()

            ep_rwd = 0.0
            while not done:
                #self.env.render()
                act = self.get_action(policy, obs)
                nobs, rwd, done, _ = self.env.step(act)
                self.add_to_buffer(obs, act, rwd, nobs, done)
                obs = nobs
                ep_rwd += rwd

            avg_rwd += ep_rwd
        avg_rwd /= episodes
        print("Reward " + str(avg_rwd))


def transform_observation(obs):
        board = obs['board']
        features = [np.isin(board, 0).astype(np.uint8), np.isin(board, 1).astype(np.uint8), np.isin(board, 2).astype(np.uint8), np.isin(board, 3).astype(np.uint8), np.isin(board, 4).astype(np.uint8),
                    np.isin(board, 6).astype(np.uint8), np.isin(board, 7).astype(np.uint8), np.isin(board, 8).astype(np.uint8), np.isin(board, 10).astype(np.uint8), np.isin(board, 11).astype(np.uint8),
                    np.isin(board, 12).astype(np.uint8), np.isin(board, 13).astype(np.uint8)]
        transformed = np.stack(features, axis=-1)
        transformed = np.moveaxis(transformed, -1, 0) #move channel dimension to front (pytorch expects this)
        return transformed


class TrainAgent(agents.BaseAgent):
    def __init__(self, policy):
        super(TrainAgent, self).__init__()
        self.policy = policy
        self.device = torch.device("cpu")

    def act(self, obs, action_space):
        obs = transform_observation(obs)
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        act = self.policy(obs)
        return act.detach().numpy()[0]


class StaticAgent(agents.BaseAgent):
    def act(self, obs, action_space):
        return 5

class DataGeneratorPommerman:
    def __init__(self):
        self.device = torch.device("cpu")

        # Define replay pool
        self.buffer = []
        self.idx = 0

        self.logger = Logger('log')

    def add_to_buffer(self, obs, act, rwd, nobs, done):
        if len(self.buffer) < replay_size:
            self.buffer.append([transform_observation(obs), act, [rwd], transform_observation(nobs), [done]])
        else:
            self.buffer[self.idx] = [transform_observation(obs), act, [rwd], transform_observation(nobs), [done]]
        self.idx = (self.idx + 1) % replay_size

    def get_batch_buffer(self, size):
        batch = list(zip(*random.sample(self.buffer, size)))
        return np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4])

    def generate(self, episodes, policy):
        # Assume first agent is TrainAgent
        agent_list = [
            TrainAgent(policy),
            agents.SimpleAgent(),
            StaticAgent(),
            StaticAgent(),
        ]
        env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

        res = np.array([0.0, 0.0, 0.0, 0.0])
        ties = 0.0
        avg_rwd = 0.0
        for i_episode in range(episodes):
            obs = env.reset()
            done = False
            ep_rwd = 0.0
            dead_before = False
            while not done:
                #env.render()
                act = env.act(obs)
                nobs, rwd, done, _ = env.step(act)

                if 10 in nobs[0]['alive']: #if train agent is still alive
                    self.add_to_buffer(obs[0], act[0], 1.0, nobs[0], False)
                    ep_rwd += 1.0
                else:
                    if not dead_before:
                        self.add_to_buffer(obs[0], act[0], 0.0, nobs[0], True)
                    dead_before = True

                #self.add_to_buffer(obs[0], act[0], rwd[0], nobs[0], done)
                obs = nobs

            avg_rwd += ep_rwd
            winner = np.where(np.array(rwd) == 1)[0]
            if len(winner) == 0:
                ties += 1
            else:
                res[winner] += 1

        avg_rwd /= episodes
        print("Wins: " + str(res) + ", Ties: " + str(ties) + ", Avg. Reward: " + str(avg_rwd))
        self.logger.write(res, ties, avg_rwd)
        env.close()
