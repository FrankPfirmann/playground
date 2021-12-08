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
        np.isin(board, Item.Agent1.value).astype(np.uint8)
    ]

    transformed = np.stack(planes, axis=-1)
    transformed = np.moveaxis(transformed, -1, 0) #move channel dimension to front (pytorch expects this)
    features['board']=transformed
    return transformed