import numpy as np

import pommerman
from agents.dqn_agent import DQNAgent
from pommerman.agents import SimpleAgent
from agents.skynet_agents import SmartRandomAgent, SmartRandomAgentNoBomb, StaticAgent
from pommerman.agents.http_agent import HttpAgent

import params as p

episodes = 100
render = False

def get_agents():
    # get policy of models
    model_dir1 = "./data/tensorboard/communicate_2/99_1" # Agent 1
    model_dir2 = "./data/tensorboard/communicate_2/99_2" # Agent 2

    # Prepare own agents
    agent1 = DQNAgent(model_dir1)
    agent2 = DQNAgent(model_dir2)
    #agent1 = HttpAgent(port=10080, host='localhost')
    #agent2 = HttpAgent(port=10081, host='localhost')

    # Prepare enemy agents
    enemy1  = StaticAgent()
    enemy2  = StaticAgent()
    #enemy1  = SimpleAgent()
    #enemy2  = SimpleAgent()
    #enemy1  = SmartRandomAgent()
    #enemy2  = SmartRandomAgent()
    #enemy1  = SmartRandomAgentNoBomb()
    #enemy2  = SmartRandomAgentNoBomb()
    #enemy1 = HttpAgent(port=10082, host='localhost'),
    #enemy2 = HttpAgent(port=10083, host='localhost'),

    if np.random.randint(2) == 0:
        return (0,[
            agent1,
            enemy1,
            agent2,
            enemy2,
        ])
    else:
        return (1, [
            enemy1,
            agent1,
            enemy2,
            agent2
        ])

wins = 0
ties = 0
losses = 0
steps_wl = []
steps_t  = []

print(f"Starting test run with {episodes} games")
for i in range(episodes):
    episode_reward = 0
    # run game
    idx, agent_list = get_agents()
    env = pommerman.make(p.env, agent_list)
    obs = env.reset()
    done = False
    step = 0
    while not done:
        if render == True:
            env.render()
        actions = env.act(obs)
        obs, reward, done, info = env.step(actions)
        episode_reward += reward[0]
        step+=1
    env.render(close=True)
    env.close()
    # count wins, ties, losses
    if 'winners' in info:
        if info['winners'][0] == idx:
            wins += 1
        else:
            losses += 1
        steps_wl.append(step)
    else:
        ties += 1
        steps_t.append(step)

    print(f"Episode {i+1}; Wins: {wins}/{100*wins/(i+1)}%; Ties: {ties}/{100*ties/(i+1)}%; Losses: {losses}/{100*losses/(i+1)}%; Steps(WL): {sum(steps_wl)/len(steps_wl) if len(steps_wl)>0 else 0}; Steps(T): {sum(steps_t)/len(steps_t) if len(steps_t)>0 else 0}")
print("Finished test!")