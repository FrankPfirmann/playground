import numpy as np

def staying_alive_reward(nobs, agent_id):
    """
    Return a reward if the agent with the given id is alive.

    :param nobs: The game state
    :param agent_id: The agent to check

    :return: The reward for staying alive
    """
    #print(nobs[0]['position'][0])
    if agent_id in nobs[0]['alive']:
        return 1.0
    else:
        return 0.0


def go_down_right_reward(nobs, high_pos, agent_num, act):
    """
    Return a reward for going to the low or right side of the board

    :param nobs: The current observation
    :param high_pos: Tuple of lowest and most-right position
    :param agent_num: The id of the agent to check (0-3)

    :return: The reward for going down or right
    """

    # only give rewards if a new highest point is reached
    bomb_bonus = 0
    if act[agent_num] == 5:
        bomb_bonus = 0.00
    if nobs[agent_num]['position'][0] > high_pos[0]:
        return 1 + bomb_bonus, (nobs[agent_num]['position'][0], high_pos[1])

    elif nobs[agent_num]['position'][1] > high_pos[1]:
        return 1 + bomb_bonus, (high_pos[0], nobs[agent_num]['position'][1])
    else:
        return 0 + bomb_bonus, high_pos
        
        
def bomb_reward(nobs, act, agent_ind):
    other_inds = [i for i in range(0, len(nobs))]
    other_inds.pop(agent_ind)
    dist = np.min([np.sqrt(np.sum(np.power(np.array(nobs[agent_ind]['position']) - np.array(nobs[ind]['position']), 2))) for ind in other_inds])
    dist = max(1.0, dist)
             
    rwd = 0.0                
    if act[agent_ind] == 5:
        rwd = 5.0/dist
    elif act[agent_ind] == 0:
        rwd = 0.0
    else:
        rwd = 1.0/dist 

    return rwd
