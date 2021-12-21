from pommerman.constants import Action

class DataAugmentor():
    """
    A class that creates new valid state transitions based on the input
    transition.
    """
    def __init__(self) -> None:
        pass

    def augment(self, obs: dict, action: Action, reward: float, nobs: dict, done: bool) -> list:
        """
        Take a state transition and create one or more new ones from it.

        The default implementation simply returns the same data it got
        in, but wrapped in a tuple and put into a list.

        :param obs: The intial state
        :param action: The action chosen by the model
        :param reward: The reward given for the transition
        :param nobs: The new state after the action was taken
        :param done: True if the episode is finished, otherwise false

        :return: A list of state transitions, where each transition is
            represented by a tuple
        """
        return [(obs, action, reward, nobs, done)]