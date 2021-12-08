from datetime import datetime
import os
import sys
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import params as p

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
    q = Pommer_Q()
    q_target = Pommer_Q()
    algo = DQN(q, q_target)
    data_generator = DataGeneratorPommerman()

    run_name=datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir=os.path.join("./data/tensorboard/", run_name)
    writer = SummaryWriter(log_dir=log_dir)

    for i in range(p.num_iterations):
        print("Iteration: " + str(i))
        policy = algo.get_policy()

        res, ties, avg_rwd = data_generator.generate(p.episodes_per_iter, policy)
        win_ratio = res[0] / (sum(res)+ties)

        total_loss=0
        for j in range(p.gradient_steps_per_iter):
            batch = data_generator.get_batch_buffer(p.batch_size)
            loss=algo.train(batch)
            total_loss+=loss
        avg_loss=total_loss/p.gradient_steps_per_iter

        writer.add_scalar('Avg. Loss/train', avg_loss, i)
        writer.add_scalar('Avg. Reward/train', avg_rwd, i)
        writer.add_scalar('Win Ratio/train', win_ratio, i)

        print("------------------------")
        if i % p.intermediate_test == p.intermediate_test-1:
            print("doing test")
            algo.set_train(False)
            policy = algo.get_policy()
            
            model_save_path = log_dir + "/" + str(i)
            torch.save(algo.q_network.state_dict(), model_save_path)
            print("Saved model to : " + model_save_path)

            data_generator.generate(p.episodes_per_iter, policy)
            algo.set_train(True)

            print("------------------------")
    writer.flush()
    writer.close()


#test_dqn('CartPole-v1')
test_pommerman_dqn()

