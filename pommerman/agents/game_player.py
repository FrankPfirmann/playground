"""
file to create a list of games to train the model with or to get test data, i.e. generate data
"""
import os
from collections import deque

import pommerman


def _calc_rewards(obs, prev_obs):
    r = [0.0, 0.0, 0.0, 0.0]
    for i in range(0,4):
        r[i] = prev_obs[i]["position"][0] - obs[i]["position"][0]
    return r


def play_games(n_games, agent_list):
    games = []
    game_n = 0
    env = pommerman.make("PommeFFACompetition-v0", agent_list)
    pos_q = deque(maxlen=64)
    # TODO: Do game playing and store the info
    while game_n < n_games:
        gamestates = []
        obs = env.reset()
        pos_q.clear()
        while True:
            my_agent = agent_list[0]
            if not my_agent.is_alive:
                print("Agent died " + str(game_n) + ". Cancelling..")
                break
            actions = []
            for i, agent in enumerate(agent_list):
                actions.append(agent.act(obs[i], env.action_space))
            state_obs, reward, game_over, _ = env.step(actions)
            rewards = _calc_rewards(state_obs, obs)

            # TODO: record only for player agents i.e., players that do training (not 0)
            # TODO: transform state observation into more fitting representation (features)
            gamestates.append((my_agent.state, actions[0]))
            if game_over:
                print("Game " + str(game_n) + " of batch has ended")
                break
            obs = state_obs
        games.append(gamestates)
        game_n += 1
    env.close()

    #return x games with y time steps and (states and actionsO)states are 11x11 bitmaps (atm 7)
    return games
