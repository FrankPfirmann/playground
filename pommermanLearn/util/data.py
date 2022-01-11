import numpy as np
from pommerman.constants import Item

def transform_observation(obs):
    """
    Transform a singular observation of the board into a stack of
    binary planes.

    :param obs: The observation containing the board

    :return: A stack of binary planes
    """
    features={}

    board = obs['board']
    planes = [
        np.isin(board, Item.Passage.value).astype(np.uint8),
        np.isin(board, Item.Rigid.value).astype(np.uint8),
        np.isin(board, Item.Wood.value).astype(np.uint8),
        np.isin(board, Item.Bomb.value).astype(np.uint8),
        np.isin(board, Item.Flames.value).astype(np.uint8),
        np.isin(board, Item.ExtraBomb.value).astype(np.uint8),
        np.isin(board, Item.IncrRange.value).astype(np.uint8),
        np.isin(board, Item.Kick.value).astype(np.uint8),
        np.isin(board, Item.Agent0.value).astype(np.uint8),
        np.isin(board, Item.Agent1.value).astype(np.uint8),
        np.isin(board, Item.Agent2.value).astype(np.uint8),
        np.isin(board, Item.Agent3.value).astype(np.uint8)
    ]

    transformed = np.stack(planes, axis=-1)
    transformed = np.moveaxis(transformed, -1, 0) #move channel dimension to front (pytorch expects this)
    features['board']=transformed
    return transformed


def _centralize_planes(planes, pos):
    b_size = planes[0].shape[0]
    central_b_size = 2 * b_size - 1
    centralized_planes = []
    for p in planes:
        central = np.zeros((central_b_size, central_b_size))
        start = ((b_size-1) - pos[0], (b_size-1) - pos[1])
        central[start[0]:start[0]+b_size, start[1]:start[1]+b_size] = p
        centralized_planes.append(central)
    return centralized_planes

def transform_observation_centralized(obs):
    """
    Transform a singular observation of the board into a stack of
    binary planes, where the agent is always in the middle of the board. Fill up non-board tiles with 0

    :param obs: The observation containing the board

    :return: A stack of binary planes
    """
    features={}

    board = obs['board']
    planes = [
        np.isin(board, Item.Passage.value).astype(np.uint8),
        np.isin(board, Item.Rigid.value).astype(np.uint8),
        np.isin(board, Item.Wood.value).astype(np.uint8),
        np.isin(board, Item.Bomb.value).astype(np.uint8),
        np.isin(board, Item.Flames.value).astype(np.uint8),
        np.isin(board, Item.ExtraBomb.value).astype(np.uint8),
        np.isin(board, Item.IncrRange.value).astype(np.uint8),
        np.isin(board, Item.Kick.value).astype(np.uint8),
        np.isin(board, Item.Agent0.value).astype(np.uint8),
        np.isin(board, Item.Agent1.value).astype(np.uint8),
        np.isin(board, Item.Agent2.value).astype(np.uint8),
        np.isin(board, Item.Agent3.value).astype(np.uint8)
    ]
    planes = _centralize_planes(planes, obs['position'])

    transformed = np.stack(planes, axis=-1)
    transformed = np.moveaxis(transformed, -1, 0) #move channel dimension to front (pytorch expects this)
    features['board']=transformed
    return transformed


def calc_dist(agent_ind, nobs, teammate_ind=-1):
    other_inds = [i for i in range(0, len(nobs))]
    other_inds.remove(agent_ind)
    if teammate_ind != -1:
        other_inds.remove(teammate_ind)
    dist = np.min([np.sqrt(np.sum(np.power(np.array(nobs[agent_ind]['position']) - np.array(nobs[ind]['position']), 2))) for ind in other_inds])
    dist = max(1.0, dist)
    return dist