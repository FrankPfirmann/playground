import gym
import torch
import numpy as np

from models import DQN_Q, Pommer_Q
from data_generator import DataGeneratorGymDiscrete, DataGeneratorPommerman
from dqn import DQN


def test_dqn(gym_env):
    num_iterations = 100
    episodes_per_iter = 10
    gradient_steps_per_iter = 1000
    batch_size = 64

    device = torch.device('cpu')
    env = gym.make(gym_env)
    data_generator = DataGeneratorGymDiscrete(env)

    q1_network = DQN_Q(env.observation_space.shape[0], env.action_space.n, 256).to(device=device)
    q1_target_network = DQN_Q(env.observation_space.shape[0], env.action_space.n, 256).to(device=device)
    algo = DQN(q1_network, q1_target_network)

    for i in range(num_iterations):
        policy = algo.get_policy()
        data_generator.generate(episodes_per_iter, policy)
        for j in range(gradient_steps_per_iter):
            batch = data_generator.get_batch_buffer(batch_size)
            algo.train(batch)

def test_pommerman_dqn():
    num_iterations = 1000000
    episodes_per_iter = 25
    gradient_steps_per_iter = 1000
    batch_size = 64
    intermediate_test = 5

    q = Pommer_Q()
    q_target = Pommer_Q()
    algo = DQN(q, q_target)
    data_generator = DataGeneratorPommerman()

    for i in range(num_iterations):
        print("Iteration: " + str(i))
        policy = algo.get_policy()
        data_generator.generate(episodes_per_iter, policy)
        for j in range(gradient_steps_per_iter):
            batch = data_generator.get_batch_buffer(batch_size)
            algo.train(batch)
        print("------------------------")
        if i % intermediate_test == intermediate_test-1:
            print("doing test")
            algo.set_train(False)
            policy = algo.get_policy()
            data_generator.generate(episodes_per_iter, policy)
            algo.set_train(True)

            print("------------------------")


#test_dqn('CartPole-v1')
test_pommerman_dqn()

