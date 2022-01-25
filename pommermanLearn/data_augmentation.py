from pommerman.constants import Action
import numpy as np

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

class DataAugmentor_v1(DataAugmentor):
    """
    A class that creates new valid state transitions based on the input
    transition using rotation, mirroring and point reflecting.
    """
    def _init_(self) -> None:
        pass

    def augment(self, obs: dict, action: int, reward: float, nobs: dict, done: bool) -> list:
        """
        Take a state transition and create rotated, mirrored and point reflected versions from it

        :param obs: The intial state
        :param action: The action chosen by the model
        :param reward: The reward given for the transition
        :param nobs: The new state after the action was taken
        :param done: True if the episode is finished, otherwise false

        :return: A list of state transitions, where each transition is
            represented by a tuple
        """
        transitions = []

        # rotate transitions by 90, 180 and 270 degrees counterclockwise
        obs_90 = self.rotate_obs(obs)
        nobs_90 = self.rotate_obs(nobs)
        action_90 = self.rotate_action(action)
        obs_180 = self.rotate_obs(obs_90)
        nobs_180 = self.rotate_obs(nobs_90)
        action_180 = self.rotate_action(action_90)
        obs_270 = self.rotate_obs(obs_180)
        nobs_270 = self.rotate_obs(nobs_180)
        action_270 = self.rotate_action(action_180)
        transitions.append((obs_90, action_90, reward, nobs_90, done))
        transitions.append((obs_180, action_180, reward, nobs_180, done))
        transitions.append((obs_270, action_270, reward, nobs_270, done))

        # mirror transitions horizontally and vertically
        obs_mirrored_hor = self.mirror_horizontal_obs(obs)
        nobs_mirrored_hor = self.mirror_horizontal_obs(nobs)
        action_mirrored_hor = self.mirror_horizontal_action(action)
        obs_mirrored_ver = self.mirror_vertical_obs(obs)
        nobs_mirrored_ver = self.mirror_vertical_obs(nobs)
        action_mirrored_ver = self.mirror_vertical_action(action)
        transitions.append((obs_mirrored_hor, action_mirrored_hor, reward, nobs_mirrored_hor, done))
        transitions.append((obs_mirrored_ver, action_mirrored_ver, reward, nobs_mirrored_ver, done))

        # point reflect transition
        obs_point_refl = self.mirror_horizontal_obs(obs_mirrored_ver)
        nobs_point_refl = self.mirror_horizontal_obs(nobs_mirrored_ver)
        action_point_refl = self.mirror_horizontal_action(action_mirrored_ver)
        transitions.append((obs_point_refl, action_point_refl, reward, nobs_point_refl, done))

        # mirror transitions diagonally
        obs_mirrored_dia_1  = self.rotate_obs(obs_mirrored_ver)
        nobs_mirrored_dia_1 = self.rotate_obs(nobs_mirrored_ver)
        action_mirrored_dia_1 = self.rotate_action(action_mirrored_ver)
        obs_mirrored_dia_2  = self.rotate_obs(obs_mirrored_hor)
        nobs_mirrored_dia_2 = self.rotate_obs(nobs_mirrored_hor)
        action_mirrored_dia_2 = self.rotate_action(action_mirrored_hor)
        transitions.append((obs_mirrored_dia_1, action_mirrored_dia_1, reward, nobs_mirrored_dia_1, done))
        transitions.append((obs_mirrored_dia_2, action_mirrored_dia_2, reward, nobs_mirrored_dia_2, done))

        return transitions

    def rotate_obs(self, obs):
        """
        Rotate an agents observation by 90 degrees counter-clockwise
        """
        rotated_obs = obs.copy()
        rotated_obs['board'] = np.rot90(obs['board'])
        rotated_obs['bomb_blast_strength'] = np.rot90(obs['bomb_blast_strength']) 
        rotated_obs['bomb_life'] = np.rot90(obs['bomb_life']) 
        rotated_obs['bomb_moving_direction'] = np.rot90(obs['bomb_moving_direction']) 
        rotated_obs['flame_life'] = np.rot90(obs['flame_life'])

        # rotate position of agent
        rotated_obs['position'] = (-obs['position'][1]+10, obs['position'][0]) 
        return rotated_obs

    def rotate_action(self, action):
        """
        Rotate an agents actions by 90 degrees counter-clockwise
        """
        if action == Action.Stop.value or action == Action.Bomb.value:
            action_rotated = action
        elif action == Action.Up.value:
            action_rotated = Action.Left.value
        elif action == Action.Down.value:
            action_rotated = Action.Right.value
        elif action == Action.Left.value:
            action_rotated = Action.Down.value
        elif action == Action.Right.value:
            action_rotated = Action.Up.value
        return action_rotated

    def mirror_horizontal_obs(self, obs):
        """
        Mirror an agents observation horizontally
        """
        mirrored_obs = obs.copy()
        mirrored_obs['board'] = np.flip(obs['board'], 1)
        mirrored_obs['bomb_blast_strength'] = np.flip(obs['bomb_blast_strength'], 1) 
        mirrored_obs['bomb_life'] = np.flip(obs['bomb_life'], 1) 
        mirrored_obs['bomb_moving_direction'] = np.flip(obs['bomb_moving_direction'], 1) 
        mirrored_obs['flame_life'] = np.flip(obs['flame_life'], 1)

        # mirror position of agent
        mirrored_obs['position'] = (10-obs['position'][0], obs['position'][1]) 
        return mirrored_obs

    def mirror_horizontal_action(self, action):
        '''
        Mirror an agents observation horizontally
        '''
        if action == Action.Stop.value or action == Action.Bomb.value or action == Action.Up.value or action == Action.Down.value:
            action_mirrored = action
        elif action == Action.Left.value:
            action_mirrored = Action.Right.value
        elif action == Action.Right.value:
            action_mirrored = Action.Left.value
        return action_mirrored

    def mirror_vertical_obs(self, obs):
        """
        Mirror an agents transition vertically
        """
        mirrored_obs = obs.copy()
        mirrored_obs['board'] = np.flip(obs['board'], 0)
        mirrored_obs['bomb_blast_strength'] = np.flip(obs['bomb_blast_strength'], 0) 
        mirrored_obs['bomb_life'] = np.flip(obs['bomb_life'], 0) 
        mirrored_obs['bomb_moving_direction'] = np.flip(obs['bomb_moving_direction'], 0) 
        mirrored_obs['flame_life'] = np.flip(obs['flame_life'], 0)

        # mirror position of agent
        mirrored_obs['position'] = (obs['position'][0], 10-obs['position'][1]) 
        return mirrored_obs

    def mirror_vertical_action(self, action):
        """
        Mirror an agents actions vertically
        """
        if action == Action.Stop.value or action == Action.Bomb.value or action == Action.Left.value or action == Action.Right.value:
            action_mirrored = action
        elif action == Action.Up.value:
            action_mirrored = Action.Down.value
        elif action == Action.Down.value:
            action_mirrored = Action.Up.value
        return action_mirrored
