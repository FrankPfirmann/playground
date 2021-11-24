"""
Main method consisting of calling game_player to generate a batch of games and then training the model with those games
"""
from game_player import play_games
from dqn_ffa_agent import DQNFFAAgent
from nn_model import CNNModel
# TODO: add parseargs


def train(epochs, n_games):
    epoch_n = 0
    nn_model = CNNModel(1)
    agent = DQNFFAAgent(nn_model)
    while epoch_n < epochs:

        games = play_games(n_games, agent)
        nn_model.optimize(games)
        epoch_n += 1
    return


if __name__ == "__main__":
    train(5, 5)
