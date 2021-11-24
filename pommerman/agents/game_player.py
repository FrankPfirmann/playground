"""
file to create a list of games to train the model with or to get test data, i.e. generate data
"""
import os

import pommerman


def play_games(n_games, agent_list):
    games = []
    game_n = 0
    env = pommerman.make("PommeFFACompetition-v0", agent_list)

    # TODO: Do game playing and store the info
    while game_n < n_games:
        gamestates = []
        obs = env.reset()

        while True:
            actions = []
            for agent in agent_list:
                actions.append(agent.act(obs, env.action_space))
            state_obs, reward, game_over, _ = env.step(actions)
            # TODO: record only for player agents i.e., players that do training (not 0)
            # TODO: transform state observation into more fitting representation (features)
            gamestates.append((state_obs[0], actions[0]))
            if game_over:
                print("Game " + str(game_n) + " of batch has ended")
                break
        games.append(gamestates)
        game_n += 1
    env.close()
    return games
