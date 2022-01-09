import numpy as np

from pommerman.constants import Item
from util.data import calc_dist


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
    dist = calc_dist(agent_ind, nobs)
    rwd = 0.0                
    if act[agent_ind] == 5:
        rwd = 5.0/dist
    elif act[agent_ind] == 0:
        rwd = 0.0
    else:
        rwd = 1.0/dist 

    return rwd


def skynet_reward(obs, act, nobs, fifo):
    """
    Skynet reward function rewarding enemy deaths, powerup pickups and stepping on blocks not in FIFO
    :param obs: previous observation
    :param nobs: new observation
    :param fifo: 121 (11x11) cell queue
    :return:
    """
    # calculate rewards for all agents
    r = [0.0] * len(obs)
    for i in range(len(obs)):
        dist = calc_dist(i, nobs)
        if i!=0:
            continue
        n_enemies_prev = 0
        alive_prev = obs[i]['alive']
        for e in obs[i]['enemies']:
            if e.value in alive_prev:
                n_enemies_prev += 1
        prev_can_kick = obs[i]['can_kick']
        prev_n_ammo = obs[i]['ammo']
        prev_n_blast = obs[i]['blast_strength']

        cur_alive = nobs[i]['alive']
        n_enemy_cur = 0
        for e in nobs[i]['enemies']:
            if e.value in cur_alive:
                n_enemy_cur += 1

        cur_can_kick = nobs[i]['can_kick']
        cur_n_ammo = nobs[i]['ammo']
        cur_n_blast = nobs[i]['blast_strength']
        cur_position = nobs[i]['position']
        if n_enemies_prev - n_enemy_cur > 0:
            r[i] += (n_enemies_prev - n_enemy_cur) * 0.5

        if not prev_can_kick and cur_can_kick:
            r[i] += 0.02

        if cur_n_ammo - prev_n_ammo > 0 and obs[i]['board'][cur_position[0]][cur_position[1]] == Item.ExtraBomb.value:
            r[i] += 0.01

        if cur_n_blast - prev_n_blast > 0:
            r[i] += 0.01

        if cur_position not in fifo[i]:
            r[i] += 0.001

        if act[i] == 5:
            r[i] += 0.05 / dist

        if len(fifo[i]) == 121:
            fifo[i].pop()
        fifo[i].append(cur_position)
    return r
