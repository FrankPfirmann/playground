import numpy as np
from pommerman.constants import Item

from pommermanLearn.util.analytics import Stopwatch


def transform_observation(obs, p_obs=False, centralized=False):
    """
    Transform a singular observation of the board into a stack of
    binary planes.

    :param obs: The observation containing the board
    :param p_obs: Whether the observation is partially observable
    :param centralized: Whether we want the fully observable board to centralize

    :return: A stack of binary planes
    """
    features = {}
    board = obs['board']
    planes = [                                                  # Index
        np.isin(board, Item.Passage.value).astype(np.uint8),    # 0
        np.isin(board, Item.Rigid.value).astype(np.uint8),      # 1
        np.isin(board, Item.Wood.value).astype(np.uint8),       # 2
        np.isin(board, Item.Bomb.value).astype(np.uint8),       # 3
        np.isin(board, Item.Flames.value).astype(np.uint8),     # 4
        np.isin(board, Item.ExtraBomb.value).astype(np.uint8),  # 5
        np.isin(board, Item.IncrRange.value).astype(np.uint8),  # 6
        np.isin(board, Item.Kick.value).astype(np.uint8),       # 7
        np.isin(board, Item.Agent0.value).astype(np.uint8),     # 8
        np.isin(board, Item.Agent1.value).astype(np.uint8),     # 9
        np.isin(board, Item.Agent2.value).astype(np.uint8),     # 10
        np.isin(board, Item.Agent3.value).astype(np.uint8)#,     # 11
        #np.isin(board, Item.Fog.value).astype(np.uint8)         # 12
    ]
    if p_obs:
        planes = _centralize_planes_partial(planes, obs['position'])
    elif centralized:
        planes = _centralize_planes(planes, obs['position'])

    transformed = np.stack(planes, axis=-1)
    transformed = np.moveaxis(transformed, -1, 0)  # move channel dimension to front (pytorch expects this)
    features['board'] = transformed
    return transformed


def _centralize_planes(planes, pos):
    b_size = planes[0].shape[0]
    central_b_size = 2 * b_size - 1
    centralized_planes = []
    for p in planes:
        central = np.zeros((central_b_size, central_b_size))
        start = ((b_size - 1) - pos[0], (b_size - 1) - pos[1])
        central[start[0]:start[0] + b_size, start[1]:start[1] + b_size] = p
        centralized_planes.append(central)
    return centralized_planes


def _centralize_planes_partial(planes, pos):
    b_size = 5
    central_b_size = 2 * b_size - 1
    partial_planes = []
    board_length = len(planes[0][0])
    for p in planes:
        partial = np.zeros((central_b_size, central_b_size))

        partial[max((b_size-1) - pos[0], 0):min(board_length - pos[0] + b_size-1, central_b_size),
        max((b_size-1) - pos[1], 0):min(board_length - pos[1] + b_size-1, central_b_size)] = \
            p[max(pos[0] - (b_size - 1), 0):min(board_length, pos[0] + (b_size)),
            max(pos[1] - (b_size - 1), 0):min(board_length, pos[1] + (b_size))]
        partial_planes.append(partial)

    outside_board = np.zeros((central_b_size, central_b_size))
    for i in range(0, central_b_size):
        for j in range(0, central_b_size):
            plane_x = (pos[0] + i - b_size + 1)
            plane_y = (pos[1] + j - b_size + 1)
            if plane_x < 0 or plane_x >= board_length or plane_y < 0 or plane_y >= board_length:
                outside_board[i][j] = 1.0

    partial_planes.append(outside_board)
    return partial_planes


def transform_observation_centralized(obs):
    return transform_observation(obs, p_obs=False, centralized=True)


def transform_observation_partial(obs):
    return transform_observation(obs, p_obs=True)


def transform_observation_simple(obs):
    return transform_observation(obs)


def calc_dist(agent_ind, nobs, teammate_ind=-1):
    other_inds = [i for i in range(0, len(nobs))]
    other_inds.remove(agent_ind)
    if teammate_ind != -1:
        other_inds.remove(teammate_ind)
    dist = np.min(
        [np.sqrt(np.sum(np.power(np.array(nobs[agent_ind]['position']) - np.array(nobs[ind]['position']), 2))) for ind
         in other_inds])
    dist = max(1.0, dist)
    return dist

def merge_views(first, second, fov, forgetfullness=0.0):
    """
    Merge the first and second view in the field of view and forget data
    outside of it.

    :param first: A numpy array respresenting the first view
    :param second: A numpy array respresenting the second view
    :param fov: A binary numpy array respresenting the field of view
    :param forgetfullness: A value ranging from 0.0 to 1.0 that reduces 
        values outside the field of view.
    """
    assert first.shape == second.shape == fov.shape, f"Shapes of planes to merge must match exactly, but first is {first.shape}, second is {second.shape} and fov is {fov.shape}"
    assert forgetfullness >= 0.0 and forgetfullness <= 1.0, "forgetfullness must be a value in the range 0.0 to 1.0"  

    remembrance=1-forgetfullness
    fog = 1-fov

    merged = second*fov + first*fog*remembrance
    return merged