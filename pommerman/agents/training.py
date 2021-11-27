"""
Main method consisting of calling game_player to generate a batch of games and then training the model with those games
"""
from game_player import play_games
from dqn_ffa_agent import DQNFFAAgent
from random_agent import RandomAgent
from pommerman import agents
from nn_model import CNNModel


def train(epochs, n_games):
    epoch_n = 0
    nn_model = CNNModel(1)

    while epoch_n < epochs:
        # load NN agent instead
        agent_list = [
            agents.DQNFFAAgent(nn_model),
            agents.RandomAgent(),
            agents.RandomAgent(),
            agents.RandomAgent()
        ]
        games = play_games(n_games, agent_list)
        nn_model.optimize(games)
        epoch_n += 1
    return

# TODO: add parseargs

if __name__ == "__main__":
    training_epochs = 1
    games_per_epoch = 3
    train(training_epochs, games_per_epoch)
