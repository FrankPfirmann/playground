"""
file to create a list of games to train the model with or to get test data, i.e. generate data
"""


def play_games(n_games, agent):
    games = []
    count = 0
    while count < n_games:
        # TODO: Do game playing and store the info
        games.append("game" + str(count+1))
        count += 1
    return games
