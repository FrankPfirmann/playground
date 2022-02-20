import logging

import numpy as np
import torch

from pommerman.constants import Item
from util.data import centralize_view, decentralize_view, merge_views, merge_views_life, merge_views_counting, \
    crop_view, create_bitmaps, merge_views_unique
import params as p


class BoardTracker:
    def __init__(self, board_size=11):
        self.board = None
        self.position = None
        self.b = board_size

    def update(self, view: np.array, position: list) -> None:
        """
        Updates the global state with the centralized and cropped ``view``

        :param view: A cropped and centralized view of the board
        :param position: The position of the views center in the global
            coordinate system.
        """
        assert view.shape[-1] == view.shape[-2]

        if self.board is None or self.board.shape != view.shape:
            self.position = position
            self.board = view
            return

        batch_size = view.shape[0]
        for sample in range(batch_size):
            for layer in range(14):  # Merge all layers but fog
                if layer in [0, 1]:  # Remember walls and passages always
                    forgetfulness = 0.0
                else:  # Forget other layers that are out of view slowly
                    forgetfulness = p.forgetfullness

                # Invert fog to get field of view

                first = self.board[sample, layer, :, :]
                second = view[sample, layer, :, :]
                fog = view[sample, -1, :, :]
                fov = 1 - fog

                first = decentralize_view(first,  self.position, (self.b, self.b))
                second = decentralize_view(second, position, (self.b, self.b))
                fov = decentralize_view(fov,    position, (self.b, self.b))

                if p.memory_method == 'forgetting':
                    if layer <= 12:
                        merged = merge_views(
                            first, second, fov, forgetfullness=forgetfulness)
                    else:
                        merged = merge_views_life(first, second, fov)
                elif p.memory_method == 'counting':
                    merged = merge_views_counting(first, second, fov)

                # Recentralize and safe into memory
                self.board[sample, layer, :, :] = centralize_view(
                    merged, position)
        self.position = position

    def get_view(self, position: list, view_range: int = 0, centralized=False) -> np.array:
        """
        Return a view of the internal global board state

        :param position: The position where the view center should lie
        :param view_range: The amount of tiles left in each direction.
            Everything outside will be cropped out if the value is
            greater then zero.
        :param centralized: Center the returned view if True, overwise
            return an uncentered board.

        :return: The resulting view
        """
        batch_size = self.board.shape[0]
        layers = self.board.shape[1]

        if view_range > 0:
            fov = 2*view_range + 1
            bounds = (batch_size, layers, fov, fov)
        else:
            bounds = self.board.shape

        if torch.is_tensor(self.board):
            views = torch.zeros(bounds, device=p.device)
        else:
            views = np.zeros(bounds)

        for sample in range(batch_size):
            for layer in range(layers):
                view = centralize_view(self.board[sample, layer, :, :], position, 0)
                #view = self.board[sample, layer, :, :]
                if view_range > 0:
                    view = crop_view(view, view_range)
                if not centralized:
                    view = decentralize_view(view, position, self.bounds)

                views[sample, layer, :, :] = view
        return views

    def reset(self) -> None:
        """
        Reset the internal representation
        """
        self.board = None
        self.position = None

    def update_memory(self, nobs, set_memory=True):
        '''
        deprecated memory update method
        :param self:
        :param nobs: the observation to update memory with
        :param set_memory: whether to return view or update memory
        :return:
        '''
        if self.memory is None or not self.validate_memory(nobs):
            self.memory = nobs
            return

        # Invert fog to get field of view
        fov = 1 - nobs[0][..., 14, :, :]

        for layer in range(14):
            if layer in [0, 1]:  # Remember walls and passages always
                forgetfulness = 0.0
            else:  # Forget other layers that are out of view slowly
                forgetfulness = p.forgetfullness
            first = self.memory[0][..., layer, :, :]
            second = nobs[0][..., layer, :, :]
            if layer <= 12:
                merged = merge_views(first, second, fov, forgetfullness=forgetfulness)
            else:
                merged = merge_views_life(first, second, fov)
            nobs[0][..., layer, :, :] = merged
        if set_memory:
            self.memory = nobs
        else:
            return nobs

    def validate_memory(self, nobs):
        if self.memory is None:
            return False

        if self.memory[0].shape != nobs[0].shape:
            return False

        if self.memory[1].shape != nobs[1].shape:
            return False
        return True


class BoardTrackerFixed:
    def __init__(self, board_size=11):
        self.board = None
        self.position = None
        self.b = board_size

    def is_none(self):
        return self.board is None

    def update(self, obs) -> None:
        """
        Updates the global state with the centralized and cropped ``view``

        :param board: A cropped and centralized view of the board
        :param position: The position of the views center in the global
            coordinate system.
        """
        board = obs['board']
        assert board.shape[-1] == board.shape[-2]
        bitmaps = np.array(create_bitmaps(board, p.centralize_planes, obs, p.p_observable, p.crop_fog))
        if self.board is None:
            self.board = bitmaps
            return

        for layer in range(14):  # Merge all layers but fog
            if layer in [0, 1]:  # Remember walls and passages always
                forgetfulness = 0.0
            else:  # Forget other layers that are out of view slowly
                forgetfulness = p.forgetfullness

            # Invert fog to get field of view
            first = self.board[layer, :, :]
            second = bitmaps[layer, :, :]
            fog = bitmaps[-1, :, :]
            fov = 1 - fog

            if p.memory_method == 'forgetting':
                if layer <= 7:
                    merged = merge_views(
                        first, second, fov, forgetfullness=forgetfulness)
                elif layer <= 11:
                    layer_alive = layer + 2 in obs['alive'] #bool whether agent belonging to layer is alive
                    merged = merge_views_unique(
                        first, second, fov, layer_alive, forgetfullness=forgetfulness)
                else:
                    merged = merge_views_life(first, second, fov)
            elif p.memory_method == 'counting':
                merged = merge_views_counting(first, second, fov)

            # Recentralize and safe into memory
            self.board[layer, :, :] = merged

    def get_view(self, position: list, centralized=False) -> np.array:
        """
        Return a view of the internal global board state

        :param position: The position where the view center should lie
        :param view_range: The amount of tiles left in each direction.
            Everything outside will be cropped out if the value is
            greater then zero.
        :param centralized: Center the returned view if True, overwise
            return an uncentered board.

        :return: The resulting view
        """
        layers = self.board.shape[0]
        assert self.board.shape[1] == self.board.shape[2]
        board_length = self.board.shape[1]
        if centralized:
            bounds = (layers, 2*board_length-1, 2*board_length-1)
        else:
            bounds = self.board.shape

        if torch.is_tensor(self.board):
            views = torch.zeros(bounds, device=p.device)
        else:
            views = np.zeros(bounds)
        for layer in range(layers):
            if centralized:
                view = centralize_view(self.board[layer, :, :], position, 0)
            else:
                view = self.board[layer, :, :]
            views[layer, :, :] = view
        return views

    def reset(self) -> None:
        """
        Reset the internal representation
        """
        self.board = None
        self.position = None

    def set_agent_spawns(self):
        '''
        set the agent spawns initially (these decay over time)
        '''
        logging.debug("set agent spawn")
        self.board[8, 1, 1] = 1.0
        self.board[9, self.b-2, 1] = 1.0
        self.board[10, self.b-2, self.b-2] = 1.0
        self.board[11, 1, self.b-2] = 1.0

    def set_teammate_pos(self, board, teammate_id, teammate_pos):
        '''
        adjusting the layer of the teammate agent
        if agent is in fov, it is already correct due to memory update, so no update necessary
        if agent not in fov, we set it to the tracked position (erase memory to not leave trail)
        :param board: board in fixed state
        :param teammate_id: teammate_id in board (e.g. 10,11,12,13)
        :param teammate_pos: position tuple that communication tracked
        '''
        if teammate_id in board:
            return
        else:
            self.board[teammate_id-2] = np.zeros((self.b, self.b))
            self.board[teammate_id-2, teammate_pos[0], teammate_pos[1]] = 1.0
            return

    def adjust_enemy_pos(self, board, enemy_id, teammate_pos, region_id):
        '''
        adjust the bitmaps for the enemy bitboards according to the tracking enemy scheme
        if less cells are elligible, the placed certainty is higher
        :param board: the fixed board the agent sees
        :param enemy_id: id of enemy (10, 11, 12, 13)
        :param teammate_pos: the tracked position of the teammate (delayed by 1)
        :param region_id: region the teammate saw the enemy based on his perspective
        '''
        if enemy_id in board:
            return
        else:
            r_id = region_id - 1  # 0 to 8
            y_dir = r_id//3 - 1
            x_dir = r_id % 3 - 1
            enemy_center = (teammate_pos[0] + y_dir*3, teammate_pos[1] + x_dir*3)
            enemy_indexes = [(i, j) for i in range(enemy_center[0] - 1, enemy_center[0]+2)
                             for j in range(enemy_center[1] - 1, enemy_center[1]+2)]

            fog = np.where(board == 5)
            fog_indices = list(zip(fog[0], fog[1]))
            # only elligible enemy spots are where fog is
            enemy_indexes = [i for i in enemy_indexes if 0 <= i[0] < self.b and 0 <= i[1] < self.b and i in fog_indices]
            self.board[enemy_id - 2] = np.zeros((self.b, self.b))
            certainty = 0 if len(enemy_indexes) == 0 else 1/len(enemy_indexes)
            for index in enemy_indexes:
                self.board[enemy_id - 2, index[0], index[1]] = certainty
