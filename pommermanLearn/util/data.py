import numpy as np

import params
from pommerman.constants import Item
import torch


def transform_observation(obs: dict, p_obs: bool=False, centralized: bool=False, crop_fog: bool=True):
    """
    Transform a singular observation of the board into a stack of
    binary planes.

    :param obs: The observation containing the board
    :param p_obs: Whether the observation is partially observable
    :param centralized: Whether we want the fully observable board to centralize

    :return: A stack of binary planes
    """
    if p_obs and not centralized:
        raise ValueError("Invalid value combination (p_obs=True and centralized=False)")

    board = obs['board']
    views = create_bitmaps(board, centralized, obs, p_obs, crop_fog)

    transformed = []
    for view in views:
        if centralized or p_obs:
            padding = 1 if view is views[-1] else 0
            view = centralize_view(view, obs['position'], padding=padding) 
        if p_obs and crop_fog:
            view = crop_view(view, view_range=4)

        transformed.append(view)
    transformed = np.stack(transformed, axis=-1)
    transformed = np.moveaxis(transformed, -1, 0)  # move channel dimension to front (pytorch expects this)
    return transformed


def get_ids_in_order(teammate_v):
    '''
    obtain the correct vlaues of agent on the board ordered by (self, teammate, lower agent, higher agent
    :param teammate_v: value of teammate on the board (e.g. 13)
    :return:
    '''
    self_v = ((teammate_v - 8) % 4) + 10
    enemy_v1 = ((teammate_v - 9) % 4) + 10
    enemy_v2 = ((teammate_v - 7) % 4) + 10
    if enemy_v2 < enemy_v1:
        enemy_v1, enemy_v2 = enemy_v2, enemy_v1
    return self_v, teammate_v, enemy_v1, enemy_v2


def create_bitmaps(board, centralized, obs, p_obs, crop_fog):
    '''
    function to create a bitmap for each possible board value
    :param board: the normal board representation
    :param centralized: whether the model expects centralized boards (e.g., 17x17)
    :param obs: observation containing additional maps flame_life and bomb_life
    :param p_obs: whether the environment is partially observable
    :return: the bitmaps representing our board
    '''
    view_type = np.float64
    self_v, teammate_v, enemy_v1, enemy_v2 = get_ids_in_order(obs['teammate'].value)
    views = [  # Index
        np.isin(board, Item.Passage.value).astype(view_type),  # 0
        np.isin(board, Item.Rigid.value).astype(view_type),  # 1
        np.isin(board, Item.Wood.value).astype(view_type),  # 2
        np.isin(board, Item.Bomb.value).astype(view_type),  # 3
        np.isin(board, Item.Flames.value).astype(view_type),  # 4
        np.isin(board, Item.ExtraBomb.value).astype(view_type),  # 5
        np.isin(board, Item.IncrRange.value).astype(view_type),  # 6
        np.isin(board, Item.Kick.value).astype(view_type),  # 7
        np.isin(board, self_v).astype(view_type),  # 8
        np.isin(board, teammate_v).astype(view_type),  # 9
        np.isin(board, enemy_v1).astype(view_type),  # 10
        np.isin(board, enemy_v2).astype(view_type),  # 11
        np.array(obs['flame_life']/3).astype(view_type),  # 12
        np.array(obs['bomb_life']/9).astype(view_type)  # 13
    ]
    if p_obs and not crop_fog:
        #normalized count of when cell was last visited
        views.append(np.ones_like(board).astype(np.uint8))  # 14
        views.append(np.isin(board, Item.Fog.value).astype(np.uint8))  # 15
    if (centralized and not p_obs) or crop_fog:
        views.append(np.zeros(board.shape))
    return views


def centralize_view(view: torch.tensor, position: np.array, padding: int=0):
    """
    Centralize the view around the given position.

    :param view: This view will be centered around ``position``.
    :param position: This position in the view will be the new center
        point
    :param padding: Areas outside the view will be padded with this
        value

    :return: The view with ``position`` as its new center point
    """

    bounds = tuple([2*d-1 for d in view.shape])
    if torch.is_tensor(view):
        centralized = torch.ones(bounds)
    else:
        centralized = np.ones(bounds)
    centralized *= padding

    ax, ay = position # Agent position
    cx, cy = calculate_center(centralized.shape)
    pw, ph = view.shape # Board width and height

    left  = cx   - ax
    right = left + pw
    up    = cy   - ay
    down  = up   + ph

    # Copy the small plane in the center of the big centralized plane 
    centralized[int(left):int(right), int(up):int(down)] = view

    return centralized


def decentralize_view(view: np.array, position: list, bounds: tuple):
    """
    Move a centralized view back to its original position

    :param view: A centralized view 
    :param position: The position in the original view
    :param bounds: A ``tuple`` containing the original view bounds in
        the form of (width, height).

    :return: The ``view`` with the center moved to ``position`` in
        ``bounds``.
    """
    ax, ay = position
    bw, bh = bounds
    cx, cy = calculate_center(view.shape)

    left  = cx   - ax
    right = left + bw
    up    = cy   - ay
    down  = up   + bh
    return view[int(left):int(right), int(up):int(down)]


def calculate_center(shape: tuple):
    """
    Calculate and return the center point of ``shape``.

    :param shape: A tuple (width, height) of odd numbers

    :return: A ``tuple`` (x, y) containing the center points coordinates
    """
    if any(d%2 == 0 for d in shape):
        raise ValueError("width and height of shape must be odd numbers")

    x, y = [int((d-1)/2) for d in shape[-2:]]
    return (x, y)


def crop_view(view: np.array, view_range: int):
    """
    Crops the view rectangularly to the given ``view_range`` around the
    views center point.
    
    :param view: The view crop. Width and height must be odd numbers.
    :param view_range: The number of columns and rows left in each
        direction after cropping, counting from the center point. Must
        be greater then zero and not exceed any view bound.

    :return: The given ``view``, but cropped up to ``view_range``
        around the center
    """
    if any(d%2 == 0 for d in view.shape):
        raise ValueError("view width and height must be odd numbers")

    if view_range <= 0:
        raise ValueError("view_range must be greater then zero")

    if view_range > min([(d-1)/2 for d in view.shape]): # Check if view range exceeds any view bound
        raise ValueError("view_range must not exceed any view bound")

    width, height = view.shape
    left  = int( (width-1) / 2 - view_range )
    right = int( width - left )
    up    = int( (height-1) / 2 - view_range )
    down  = int( height - up )

    return view[left:right, up:down]


def transform_observation_centralized(obs):
    return transform_observation(obs, p_obs=False, centralized=True)


def transform_observation_partial(obs):
    return transform_observation(obs, p_obs=True, centralized=True)


def transform_observation_simple(obs):
    return transform_observation(obs)


def transform_observation_partial_uncropped(obs):
    return transform_observation(obs, p_obs=True, centralized=True, crop_fog=False)


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

def merge_views(first: np.array, second: np.array, fov: np.array, forgetfullness: float=0.0):
    """
    Merge the first and second view in the field of view and forget data
    outside of it.

    :param first: A numpy array respresenting the first view
    :param second: A numpy array respresenting the second view
    :param fov: A binary numpy array respresenting the field of view
    :param forgetfullness: A value ranging from 0.0 to 1.0 that reduces
        values outside the field of view.

    :return: ``first`` and ``second`` merged with ``forgetfullness``
        applied outside of `fov`.
    """
    assert first.shape == second.shape == fov.shape, f"Shapes of planes to merge must match exactly, but first is {first.shape}, second is {second.shape} and fov is {fov.shape}"
    assert forgetfullness >= 0.0 and forgetfullness <= 1.0, "forgetfullness must be a value in the range 0.0 to 1.0"

    remembrance=1-forgetfullness
    fog = 1-fov

    merged = second*fov + first*fog*remembrance
    return merged


def merge_views_life(first: np.array, second: np.array, fov: np.array):
    """
        Merge the first and second view in the field of view and decrease
        the data around it by 1 (for bomb and flame life)

        :param first: A numpy array respresenting the first view
        :param second: A numpy array respresenting the second view
        :param fov: A binary numpy array respresenting the field of view

        :return: second in area of fov and first decreased by 1 for the remainder
        """
    assert first.shape == second.shape == fov.shape, f"Shapes of planes to merge must match exactly, but first is {first.shape}, second is {second.shape} and fov is {fov.shape}"
    fog = 1 - fov
    first_dec = first * fog - 1
    first_dec[first_dec < 0] = 0

    merged = second * fov + first_dec * fog
    return merged


def merge_views_unique(first: np.array, second: np.array, fov: np.array, layer_alive:bool, forgetfullness: float=0.0):
    """
        Merge the first and second view in the field of view and if a value is non-zero in
        the fov, ignore the second view (since the value is unique)
        also, set all values to 0 if respective agent is dead

        :param first: A numpy array respresenting the first view
        :param second: A numpy array respresenting the second view
        :param fov: A binary numpy array respresenting the field of view

        :return: second in area of fov and first decreased by 1 for the remainder
        """
    assert first.shape == second.shape == fov.shape, f"Shapes of planes to merge must match exactly, but first is {first.shape}, second is {second.shape} and fov is {fov.shape}"
    if not layer_alive:
        return 0*first #set layer to all zero since agent is not alive
    not_seen = not np.any(second)
    remembrance=1-forgetfullness
    fog = 1-fov

    merged = second*fov + first*fog*remembrance*not_seen
    return merged

def merge_views_counting(first: np.array, second: np.array, fov: np.array):
    """
    Merge ``first`` and ``second`` by counting the number of steps that
    elements have not been updated.

    :param first: A numpy array respresenting the first view
    :param second: A numpy array respresenting the second view
    :param fov: A binary numpy array respresenting the field of view
        values outside the field of view.

    :return: ``first`` and ``second`` merged by incrementing everything
        inside the fog and copying everything inside the fov.
    """
    assert first.shape == second.shape == fov.shape, f"Shapes of planes to merge must match exactly, but first is {first.shape}, second is {second.shape} and fov is {fov.shape}"

    fog = 1-fov

    mask = first != 0 
    inc = 1-fov
    inc[mask == False] = 0

    merged = second * fov + (first + inc)*fog
    return merged

def update_cell_visit(first: np.array, position):
    """
    Merge ``first`` and ``second`` by counting the number of steps that
    elements have not been updated.

    :param first: A numpy array respresenting the first view
    :param second: A numpy array respresenting the second view
    :param fov: A binary numpy array respresenting the field of view
        values outside the field of view.

    :return: ``first`` and ``second`` merged by incrementing everything
        inside the fog and copying everything inside the fov.
    """
    first_dec = first + 1/params.fifo_size
    first_dec[first_dec > 1] = 1
    first_dec[position[0], position[1]] = 0
    return first_dec