import numpy as np
from pommerman.constants import Item

def transform_observation(obs: dict, p_obs: bool=False, centralized: bool=False):
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
    view_type=np.float64
    views = [                                                    # Index
        np.isin(board, Item.Passage.value).astype(view_type),    # 0
        np.isin(board, Item.Rigid.value).astype(view_type),      # 1
        np.isin(board, Item.Wood.value).astype(view_type),       # 2
        np.isin(board, Item.Bomb.value).astype(view_type),       # 3
        np.isin(board, Item.Flames.value).astype(view_type),     # 4
        np.isin(board, Item.ExtraBomb.value).astype(view_type),  # 5
        np.isin(board, Item.IncrRange.value).astype(view_type),  # 6
        np.isin(board, Item.Kick.value).astype(view_type),       # 7
        np.isin(board, Item.Agent0.value).astype(view_type),     # 8
        np.isin(board, Item.Agent1.value).astype(view_type),     # 9
        np.isin(board, Item.Agent2.value).astype(view_type),     # 10
        np.isin(board, Item.Agent3.value).astype(view_type),     # 11
        np.array(obs['flame_life']).astype(view_type),           # 12
        np.array(obs['bomb_life']).astype(view_type)             # 13
    ]
    if centralized and p_obs:
        views.append(np.isin(board, Item.Fog.value).astype(np.uint8)) # 14
    
    if centralized and not p_obs:
        views.append(np.zeros(board.shape))
    
    transformed = []
    for view in views:
        if centralized or p_obs:
            padding = 1 if view is views[-1] else 0
            view = centralize_view(view, obs['position'], padding=padding) 
        if p_obs:
            view = crop_view(view, view_range=4)

        transformed.append(view)

    transformed = np.stack(transformed, axis=-1)
    transformed = np.moveaxis(transformed, -1, 0)  # move channel dimension to front (pytorch expects this)
    return transformed

def centralize_view(view: np.array, position: np.array, padding: int=0):
    """
    Centralize the view around the given position.

    :param view: This view will be centered around position. Must have
        an odd width and height.
    :param position: This position in the view will be the new center
        point
    :param padding: Areas outside the view will be padded with this
        value
    """
    if not view.shape[0]%2 and not view.shape[0]%2:
        raise ValueError("view width and height must be odd numbers")

    centralized = np.ones(tuple([2*d-1 for d in view.shape])) * padding

    ax, ay = position # Agent position
    cx, cy = [int((d-1)/2) for d in centralized.shape] # Center position
    pw, ph = view.shape # Board width and height

    left  = cx   - ax
    right = left + pw
    up    = cy   - ay
    down  = up   + ph

    # Copy the small plane in the center of the big centralized plane 
    centralized[left:right, up:down] = view

    return centralized
    
def crop_view(view: np.array, view_range: int):
    """
    Crops the view rectangularly to the given view_range around the
    views center point.
    
    :param view: The view crop. Width and height must be odd numbers.
    :param view_range: The number of columns and rows left in each
        direction after cropping, counting from the center point. Must
        be greater then zero and not exceed any view bound.
    """
    if not view.shape[0]%2 or not view.shape[1]%2:
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
    """
    assert first.shape == second.shape == fov.shape, f"Shapes of planes to merge must match exactly, but first is {first.shape}, second is {second.shape} and fov is {fov.shape}"
    assert forgetfullness >= 0.0 and forgetfullness <= 1.0, "forgetfullness must be a value in the range 0.0 to 1.0"

    remembrance=1-forgetfullness
    fog = 1-fov

    merged = second*fov + first*fog*remembrance
    return merged
