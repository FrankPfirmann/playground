import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import sys
# import pybullet_envs

from PPO import PPO
from agents.static_agent import StaticAgent
from pommerman.agents import SimpleAgent

from agents.skynet_agents import SmartRandomAgent, SmartRandomAgentNoBomb
from agents.train_agent import TrainAgent
from util.data import transform_observation_simple, transform_observation_partial, transform_observation_centralized
from util.rewards import staying_alive_reward, go_down_right_reward, bomb_reward, skynet_reward, woods_close_to_bomb_reward, dist_to_enemy_reward

import pommerman


################################### Training ###################################

def train():

    print("============================================================================================")


    ####### initialize environment hyperparameters ######

    env_name = "PommeRadioCompetition-v2"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    max_training_iterations = 1000
    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = 10          # save model frequency (in num timesteps)
    episodes_per_iter = 10
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    #####################################################


    ## Note : print/log frequencies should be > than max_ep_len


    ################ PPO hyperparameters ################

    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.01       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)

    #####################################################

    state_dim = 388

    action_dim = 6

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)


    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)


    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    #####################################################


    ################### checkpointing ###################

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    #####################################################


    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

    else:
        print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent1 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    ppo_agent2 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    ppo_agents = [ppo_agent1, ppo_agent2]
    
    agent_ind = np.random.randint(2) 
    agent_ind = 0

    agent_list1 = [
                TrainAgent(ppo_agent1, algo="ppo2"),
                SmartRandomAgentNoBomb(),
                TrainAgent(ppo_agent2, algo="ppo2"),
                SmartRandomAgentNoBomb(),
                ]

    agent_list2 = [
                SmartRandomAgentNoBomb(),
                TrainAgent(ppo_agent1, algo="ppo2"),
                SmartRandomAgentNoBomb(),
                TrainAgent(ppo_agent2, algo="ppo2")
                ]
    
    # get indices of agents
    agent_inds = [0+agent_ind, 2+agent_ind]
    enemy_inds = [x for x in range(4) if x not in agent_inds]

    agent_ids = [10+agent_ind, 12+agent_ind]

    agent_list = agent_list1 if agent_ind == 0 else agent_list2


    print("training environment name : " + env_name)

    env = pommerman.make(env_name, agent_list)
    # state space dimension
    


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")


    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')


    # printing and logging variables
    print_running_reward = []
    print_running_episodes = 0
    print_running_steps = []

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    fifo = [[] for _ in range(4)]
    skynet_reward_log = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
    res = np.array([0.0] * 2)
    ties = 0


    for i in range(4):
        fifo[i].clear()

    # training loop
    for i in range(1, max_training_iterations):

        for _ in range(episodes_per_iter):

            obs = env.reset()
            current_ep_reward = 0

            agt_rwd = 0

            for t in range(1, max_ep_len+1):
                
                act = env.act(obs)

                actions = [act[0], act[1], act[2], act[3]]

                actions[agent_inds[0]] = actions[agent_inds[0]][1].item()
                actions[agent_inds[1]] = actions[agent_inds[1]][1].item()
                nobs, reward, done, _ = env.step(actions)

                if i == 10:
                    env.render()

                skynet_rwds = skynet_reward(obs, act, nobs, fifo, agent_inds, skynet_reward_log)

                obs = nobs

                for m in range(2):

                    agt_rwd = skynet_rwds[agent_inds[m]]

                    if act[agent_inds[m]][1] == 5:
                        agt_rwd += dist_to_enemy_reward(obs, agent_inds[m], enemy_inds)

                    winner = np.where(np.array(reward) == 1)[0]
                    if done:
                        if agent_inds[m] in winner:
                            agt_rwd += 10
                    # if act[agent_inds[m]][1] == 5:
                    #     agt_rwd += woods_close_to_bomb_reward(obs, agent_inds[m])

                    ppo_agents[m].buffer.states.append(act[agent_inds[m]][0])
                    ppo_agents[m].buffer.actions.append(act[agent_inds[m]][1])
                    ppo_agents[m].buffer.logprobs.append(act[agent_inds[m]][2])
                    ppo_agents[m].buffer.rewards.append(agt_rwd)
                    ppo_agents[m].buffer.is_terminals.append(done)

                    current_ep_reward += agt_rwd
        

                time_step +=1


                if done or t == max_ep_len:

                    print_running_steps.append(t)
                    print_running_reward.append(current_ep_reward)

                    winner = np.where(np.array(reward) == 1)[0]
                    if len(winner) == 0:
                        ties += 1
                    else:
                        k = True
                        for j in range(2):
                            if agent_inds[j] in winner:
                                res[0] += 1
                                k = False
                                break
                        if k:
                            res[1] += 1

                    break


                # update PPO agent
                # if time_step % update_timestep == 0:

        ppo_agent1.update()
        ppo_agent2.update()

        print_avg_steps = int(sum(print_running_steps)/(len(print_running_steps) + 1))
        print_avg_reward = sum(print_running_reward)/(len(print_running_reward) + 1)

        print(f"Iteration : {i}, Wins: {res}, Ties: {ties}, Avg. Reward: {print_avg_reward}, Avg. Steps: {print_avg_steps}")

        print_running_reward = []
        print_running_steps = []

        res = np.array([0.0] * 2)
        ties = 0



        # save model weights
        if i % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent1.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

                # break; if the episode is over


    log_f.close()
    env.close()




    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")




if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
    
