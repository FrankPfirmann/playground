"""
Neural model to calculate the Q-Values
"""

# TODO: Add model including layers, both passes and feature transformation
import numpy as np


class CNNModel:
    def __init__(self, param):
        self.param = param

    def optimize(self, games):
        print(games[0][0][0][0])

