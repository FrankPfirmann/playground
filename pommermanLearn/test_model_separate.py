# solve error #15
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from datetime import datetime
import os
import torch
import numpy as np
import params as p

from models import Pommer_Q
from dqn import DQN
from util.data import transform_observation, transform_observation_centralized, transform_observation_partial
import params as p

import pommerman
from pommerman.agents import SimpleAgent
from agents.train_agent import TrainAgent
from agents.skynet_agents import SmartRandomAgent, SmartRandomAgentNoBomb, StaticAgent

episodes_per_iter = 50
# model_dir = './data/tensorboard/20220116T161714/29'


model_dir1 = "./data/tensorboard/20220122T231342/25992"
model_dir2 = "./data/tensorboard/20220122T231342/25991"

# against static agent
model_dir1 = './data/tensorboard/20220124T001803/1999_1'
model_dir2 = './data/tensorboard/20220124T001803/1999_2'

# static agent the smartrandom
model_dir1 = './data/tensorboard/20220124T140613/299_1'
model_dir2 = './data/tensorboard/20220124T140613/299_2'

# no agmentation, random starting position, wood + skynet reward
model_dir1 = './data/tensorboard/20220125T123909/599_1'
model_dir2 = './data/tensorboard/20220125T123909/599_2'

# new architecture, no agmentation, random starting position, wood + skynet reward, static agent
model_dir1 = './data/tensorboard/20220125T215024/999_1'
model_dir2 = './data/tensorboard/20220125T215024/999_2'

# '---', against SmartRandom, without wood rwd, gradient_steps=10, batch_size=16
model_dir1 = './data/tensorboard/20220126T121652/949_1'
model_dir2 = './data/tensorboard/20220126T121652/949_2'

# '---', prio replay
model_dir1 = './data/tensorboard/20220126T192642/249_1'
model_dir2 = './data/tensorboard/20220126T192642/249_2'

# '---', against SmartRandom, without wood rwd, gradient_steps=100, batch_size=16
# '---', prio replay
model_dir1 = './data/tensorboard/20220127T004453/499_1'
model_dir2 = './data/tensorboard/20220127T004453/499_2'


render = False
render = True
enemy = SmartRandomAgent

# get policy of models
q1 = Pommer_Q(p.p_observable, transform_observation_partial)
q_target1 = Pommer_Q(p.p_observable, transform_observation_partial)
q1.load_state_dict(torch.load(model_dir1))
q_target1.load_state_dict(torch.load(model_dir1))
algo1 = DQN(q1, q_target1, p.exploration_noise)
algo1.set_train(False)
policy1 = algo1.get_policy()

q2 = Pommer_Q(p.p_observable, transform_observation_partial)
q_target2 = Pommer_Q(p.p_observable, transform_observation_partial)
q2.load_state_dict(torch.load(model_dir2))
q_target2.load_state_dict(torch.load(model_dir2))
algo2 = DQN(q2, q_target2, p.exploration_noise)
algo2.set_train(False)
policy2 = algo2.get_policy()


# setup env
agent_list = [
    TrainAgent(policy1),
    enemy(),
    TrainAgent(policy2),
    enemy(),

]
env = pommerman.make('PommeTeamCompetition-v0', agent_list)

wins = 0
ties = 0
losses = 0

for i in range(100):
    print(i)
    episode_reward = 0
    # run game
    obs = env.reset()
    done = False
    while not done:
        if render == True:
            env.render()
        actions = env.act(obs)
        obs, reward, done, info = env.step(actions)
        episode_reward += reward[0]
    env.render(close=True)
    env.close()
    # count wins, ties, losses
    if 'winners' in info:
        if info['winners'][0] == 0:
            wins += 1
        else:
            losses += 1
    else:
        ties += 1

    print(wins/(i+1))
    print(ties/(i+1))
    print(losses/(i+1))
    print(episode_reward)

print(wins/100)
print(ties/100)
print(losses/100)