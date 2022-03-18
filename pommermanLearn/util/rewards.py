import logging

import numpy as np

from pommerman import constants
from pommerman.constants import Item
from util.data import calc_dist
import params as p

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


def skynet_reward(obs, act, nobs, fifo, agent_inds, log, done, bomb_tracker):
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
        own_id = ((obs[i]['teammate'].value - 8) % 4)+ 10
        enemy_1 = ((obs[i]['teammate'].value - 9) % 4)+ 10
        enemy_2 = ((obs[i]['teammate'].value - 7) % 4)+ 10
        alive_agents = nobs[i]['alive']
        teammate_ind = i + 2 if log_ind == 0 else i - 2
        enemies_prev = []
        alive_prev = obs[i]['alive']
        for e in obs[i]['enemies']:
            if e.value in alive_prev:
                enemies_prev.append(e.value)

        prev_n_teammate = 1 if obs[i]['teammate'].value in alive_prev else 0
        prev_can_kick = obs[i]['can_kick']
        prev_n_ammo = obs[i]['ammo']
        prev_n_blast = obs[i]['blast_strength']

        cur_alive = nobs[i]['alive']
        enemy_cur = []
        for e in nobs[i]['enemies']:
            if e.value in cur_alive:
                enemy_cur.append(e.value)
        dead_enemies = [item for item in enemies_prev if item not in enemy_cur]
        cur_n_teammate = 1 if nobs[i]['teammate'].value in cur_alive else 0
        cur_can_kick = nobs[i]['can_kick']
        cur_n_ammo = nobs[i]['ammo']
        cur_n_blast = nobs[i]['blast_strength']
        cur_position = nobs[i]['position']
        killed = False
        died = False
        tracker_items = bomb_tracker.get_killers(obs, nobs)
        if p.bomb_tracker:
            for kill in tracker_items:
                if kill['killed_agent'] + 10 not in nobs[i]['alive'] and i in kill['killers']:
                    if kill['killed_agent'] + 10 == obs[i]['teammate'].value:
                        teammate_rwd = p.teamkill_rwd
                        r[i] += teammate_rwd
                        logging.info(f"Teamkill by agent {i} rewarded with {teammate_rwd}")
                        log[log_ind][0] += teammate_rwd
                    elif not kill['killed_agent'] + 10 == own_id:
                        kill_rwd = p.kill_rwd
                        killed = True
                        r[i] += kill_rwd
                        logging.info(f"Kill by agent {i} rewarded with {kill_rwd}")
                        log[log_ind][0] += kill_rwd
        else:
            if len(dead_enemies) > 0:
                kill_rwd = p.kill_rwd
                killed = True
                r[i] += kill_rwd * len(dead_enemies)
                logging.info(f"Kill rewarded with {kill_rwd} for agent {i}")
                log[log_ind][0] += kill_rwd
            if prev_n_teammate > cur_n_teammate:
                teammate_rwd = p.teamkill_rwd
                r[i] += teammate_rwd
                logging.info(f"Teammate of agent {i} died and was rewarded with {teammate_rwd}")
                log[log_ind][4] += teammate_rwd

        if own_id in obs[i]['alive'] and own_id not in nobs[i]['alive']:
            died = True
            death_rwd = p.death_rwd
            r[i] += death_rwd
            logging.info(f"Death of agent {i} rewarded with {death_rwd}")
            log[log_ind][4] += death_rwd

        if done:
            if killed and (own_id in alive_agents or nobs[i]['teammate'].value in alive_agents) \
                    and not(enemy_1 in alive_agents or enemy_2 in alive_agents):
                win_rwd = p.win_loss_bonus
                r[i] += win_rwd
                logging.info(f"Winning blow rewarded by an extra {win_rwd} for agent {i}")
                log[log_ind][3] += win_rwd
            elif died and (enemy_1 in alive_agents or enemy_2 in alive_agents) \
                    and not (own_id in alive_agents or nobs[i]['teammate'].value in alive_agents):
                loss_rwd = -p.win_loss_bonus
                r[i] += loss_rwd
                logging.info(f"Losing death rewarded by an extra with {loss_rwd} for agent {i}")
                log[log_ind][3] += loss_rwd
        if prev_n_teammate - cur_n_teammate > 0:
            r[i] -= (prev_n_teammate-cur_n_teammate)*0.0
            log[log_ind][4] -= (prev_n_teammate-cur_n_teammate)*0.0
        item_reward = p.item_rwd
        step_reward = p.step_rwd
        if not prev_can_kick and cur_can_kick:
            r[i] += item_reward
            log[log_ind][1] += item_reward
        if cur_n_ammo - prev_n_ammo > 0 and obs[i]['board'][cur_position[0]][cur_position[1]] == Item.ExtraBomb.value:
            r[i] += item_reward
            log[log_ind][1] += item_reward
        if cur_n_blast - prev_n_blast > 0:
            r[i] += item_reward
            log[log_ind][1] += item_reward
        if cur_position not in fifo[i]:
            r[i] += step_reward
            log[log_ind][2] += step_reward
        if len(fifo[i]) == p.fifo_size:
            fifo[i].pop(0)
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
    # for each wood close to bomb reward x
    reward = (0.01 * woods_in_range) + (0.05 * enemies_in_range)
    return reward
