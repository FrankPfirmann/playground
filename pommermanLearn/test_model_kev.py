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

episodes_per_iter = 100
# model_dir = './data/tensorboard/20220116T161714/29'
# model_dir = './data/tensorboard/20220116T200110/9'

# model_dir = './data/tensorboard/20220116T204319/19'
# model_dir = './data/tensorboard/20220117T085435/139'

# almost doesn't lay bombs at all but seems to be the best rn (later steps added more often)
model_dir = './data/tensorboard/20220117T124414/9' 

# wood reward, against SmartRandom, 
model_dir = './data/tensorboard/20220118T083357/499'
model_dir = './data/tensorboard/20220118T120245/699'

# wood reward against StaticAgent
# consistently kills only one single static enemy
model_dir = './data/tensorboard/20220118T150333/199'

# best so far: only skynet rewards
# model_dir = './data/tensorboard/20220118T181847/9'
# model_dir = './data/tensorboard/20220118T181847/99'

# skynet reward against StaticAgent
# model_dir = './data/tensorboard/20220119T004707/949'
# with augmentor
model_dir = './data/tensorboard/20220119T032625/999'

# with augmentor, 200 gradient steps, batch size 32
model_dir = './data/tensorboard/20220119T111628/799'

# model_dir = "./data/tensorboard/20220121T111020/249"

render = False
# render = True
enemy = StaticAgent

# get policy of model
q = Pommer_Q(p.p_observable, transform_observation_partial)
q.load_state_dict(torch.load(model_dir))
q_target = Pommer_Q(p.p_observable, transform_observation_partial)
q_target.load_state_dict(torch.load(model_dir))
algo = DQN(q, q_target)
# algo.set_train(False)
policy = algo.get_policy()

# setup env
agent_list = [    
    # TrainAgent(policy),
    SmartRandomAgent(),
    enemy(),
    # TrainAgent(policy),
    SmartRandomAgent(),
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