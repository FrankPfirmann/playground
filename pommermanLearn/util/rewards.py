import numpy as np

from pommerman import constants
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


def skynet_reward(obs, act, nobs, fifo, agent_inds, log):
    """
    Skynet reward function rewarding enemy deaths, powerup pickups and stepping on blocks not in FIFO
    :param obs: previous observation
    :param nobs: new observation
    :param fifo: 121 (11x11) cell queue
    :return:
    """
    # calculate rewards for player agents, rest are zero
    r = [0.0] * len(obs)
    for i in range(len(obs)):
        if i not in agent_inds:
            continue
        log_ind = 0 if i <= 1 else 1
        teammate_ind = i + 2 if log_ind == 0 else i - 2
        n_enemies_prev = 0
        alive_prev = obs[i]['alive']
        for e in obs[i]['enemies']:
            if e.value in alive_prev:
                n_enemies_prev += 1

        prev_n_teammate = 1 if obs[i]['teammate'].value in alive_prev else 0
        prev_can_kick = obs[i]['can_kick']
        prev_n_ammo = obs[i]['ammo']
        prev_n_blast = obs[i]['blast_strength']

        cur_alive = nobs[i]['alive']
        n_enemy_cur = 0
        for e in nobs[i]['enemies']:
            if e.value in cur_alive:
                n_enemy_cur += 1

        cur_n_teammate = 1 if nobs[i]['teammate'].value in cur_alive else 0
        cur_can_kick = nobs[i]['can_kick']
        cur_n_ammo = nobs[i]['ammo']
        cur_n_blast = nobs[i]['blast_strength']
        cur_position = nobs[i]['position']
        if n_enemies_prev - n_enemy_cur > 0:
            r[i] += (n_enemies_prev - n_enemy_cur) * 0.5
            log[log_ind][0] += (n_enemies_prev - n_enemy_cur) * 0.5
        if prev_n_teammate - cur_n_teammate > 0:
            r[i] -= (prev_n_teammate-cur_n_teammate)*0.5
            log[log_ind][4] -= (prev_n_teammate-cur_n_teammate)*0.5
        if not prev_can_kick and cur_can_kick:
            r[i] += 0.02
            log[log_ind][1] += 0.02
        if cur_n_ammo - prev_n_ammo > 0 and obs[i]['board'][cur_position[0]][cur_position[1]] == Item.ExtraBomb.value:
            r[i] += 0.01
            log[log_ind][1] += 0.01
        if cur_n_blast - prev_n_blast > 0:
            r[i] += 0.01
            log[log_ind][1] += 0.01
        if cur_position not in fifo[i]:
            r[i] += 0.001
            log[log_ind][2] += 0.001
        if len(fifo[i]) == 121:
            fifo[i].pop()
        fifo[i].append(cur_position)
    return r


def _get_positions(board, value):
    wood_bitmap = np.isin(board, value).astype(np.uint8)
    wood_positions = np.where(wood_bitmap == 1)
    return list(zip(wood_positions[0], wood_positions[1]))


def woods_close_to_bomb_reward(obs, bomb_pos, blast_strength, agent_ids):
    '''
    :param obs: observation
    :param bomb_pos: position bomb is layed
    :param blast_strength: current blast strength of the agent
    :param agent_ids: agent ids of teammates
    :return: reward for laying bombs near wood and enemies
    '''

    board = obs['board']
    wood_positions = _get_positions(board, constants.Item.Wood.value)
    rigid_positions = _get_positions(board, constants.Item.Rigid.value)
    enemy_ids = [10,11,12,13]
    for id in agent_ids:
        enemy_ids.remove(id)
    enemy_positions =[]
    for e in enemy_ids:
        enemy_positions += _get_positions(board, e)
    woods_in_range = 0.0
    enemies_in_range = 0.0
    # for every wooden block check if it would be destroyed
    left_pos = np.asarray(bomb_pos)
    for i in range(1, blast_strength+1):
        if left_pos[0] == 0:
            break
        left_pos = (bomb_pos[0] - i, bomb_pos[1])
        if left_pos in rigid_positions:
            break
        elif left_pos in enemy_positions:
            enemies_in_range +=1
            break
        elif left_pos in wood_positions:
            woods_in_range += 1
            break
    right_pos = np.asarray(bomb_pos)
    for i in range(1, blast_strength + 1):
        if right_pos[0] == len(board)-1:
            break
        right_pos = (bomb_pos[0] + i, bomb_pos[1])
        if right_pos in rigid_positions:
            break
        elif right_pos in enemy_positions:
            enemies_in_range += 1
            break
        elif right_pos in wood_positions:
            woods_in_range += 1
            break
    down_pos = np.asarray(bomb_pos)
    for i in range(1, blast_strength + 1):
        if down_pos[1] == 0:
            break
        down_pos = (bomb_pos[0], bomb_pos[1] - i)
        if down_pos in rigid_positions:
            break
        elif down_pos in enemy_positions:
            enemies_in_range += 1
            break
        elif down_pos in wood_positions:
            woods_in_range += 1
            break
    up_pos = np.asarray(bomb_pos)
    for i in range(1, blast_strength + 1):
        if up_pos[1] == len(board)-1:
            break
        up_pos = (bomb_pos[0], bomb_pos[1] + i)
        if up_pos in rigid_positions:
            break
        elif up_pos in enemy_positions:
            enemies_in_range += 1
            break
        elif up_pos in wood_positions:
            woods_in_range += 1
            break
    # for each wood close to bomb reward 0.1
    reward = (0.002 * woods_in_range) + (0.005 * enemies_in_range)
    return reward

def dist_to_enemy_reward(obs, agent_ind, enemy_inds):
    '''
    :param obs: observation
    :param agent_ind: index of agent
    :param enemy_inds: indices of enemies
    :return: reward for laying bombs near enemies
    '''
    ag_obs = obs[agent_ind]
    board = ag_obs['board']
    pos = np.asarray(ag_obs['position'])
    enemies_pos = np.asarray([obs[enemy_inds[0]]['position'], obs[enemy_inds[1]]['position']])

    enemy_ids = [e.value for e in ag_obs['enemies'][:-1]]

    min_dist = np.inf
    for e in enemies_pos:

        dist = np.linalg.norm(e - pos)
        min_dist = dist if dist < min_dist else min_dist

    return 0.001*(1/(min_dist + 1)**2)