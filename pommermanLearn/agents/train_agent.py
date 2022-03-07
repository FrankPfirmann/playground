import logging
import random
from collections import deque

import params
from action_prune import get_filtered_actions
from pommerman import agents
from pommerman.constants import Action
from util.board_tracker import BoardTracker, BoardTrackerFixed
from util.data import centralize_view, crop_view, merge_views, merge_views_life
import numpy as np
import params as p


def _region_of_enemy(view, id):
    '''
    determine the 3x3 where the id is located (0 top left, 8 bottom right, 3 middle left etc.)
    :param view: a cropped and centralized board representation
    :param id: the agent id to search
    :return: a region id
    '''

    index = np.where(view == id)
    return np.floor(index[0][0]/3)*3 + np.floor(index[1][0]/3)


def _serialize_msg(pos, enemy):
    '''
    transform the message information into the appropriate type (base8)
    :param pos: position message from 0 to 4 (see pos_change dict)
    :param enemy: region of current enemy from 0 (not seen) to 9
    :return: message tuple to send
    '''
    message = pos * 10 + enemy
    message2 = message % 8
    message1 = np.floor(message / 8)
    return int(message1), int(message2)


def _deserialize_msg(msg):
    '''
    transform the base8 message format back to the base 10 format (position (5 values), region (10 values))
    :param msg: message to deserialize
    :return: both messages that were transmitted
    '''
    msg_total = msg[0] * 8 + msg[1]
    pos_change = np.floor(msg_total/10)
    enemy_msg = msg_total % 10
    return pos_change, enemy_msg


class TrainAgent(agents.BaseAgent):
    """
    An agent used for training a pytorch model 
    """

    def __init__(self, policy, transformer, communicate=params.communicate, is_train=True):
        """
        Initializes the agent and sets the model to train 

        :param policy: The pytorch model to train
        """
        super(TrainAgent, self).__init__()
        self.policy = policy

        bsize = 9 if p.env == "custom-v2" else 11
        self.init = True
        self.prev_pos = None
        self.teammate_pos = None
        self.enemy_ids = None
        # bit signaling which agents position should be transmitted
        self.transmit_first = True
        self.memory = BoardTrackerFixed(board_size=bsize)
        self.is_train = is_train # if this is true memory has to be updated in the act method instead of data_generator
        self.obs = None # save transformed observation to avoid additional centralization
        self.next_msg = None # saved message to transmit
        self.pos_change = {(0, 0): Action.Stop.value,
                           (-1, 0): Action.Up.value,
                           (1, 0): Action.Down.value,
                           (0, 1): Action.Right.value,
                           (0, -1): Action.Left.value}

        self.pos_change_effect = {v: k for k, v in self.pos_change.items()}
        self.teammate_start = {0: (bsize-2, bsize-2),
                               1: (1, bsize-2),
                               2: (1, 1),
                               3: (bsize-2, 1)}
        self.communicate = communicate
        self.transformer = transformer

    def get_memory_view(self):
        '''
        :return: transformed saved observation
        '''
        return self.obs

    def update_memory(self, obs):
        '''
        update the fixed board memory, then transform it to the view the model expects
        :param obs: observation dict the agent receives
        '''
        self.memory.update(obs)
        if self.communicate:
            self.do_communication(obs)
        views = self.memory.get_view(obs["position"], centralized=p.centralize_planes)
        self.obs = self.transformer(obs, pre_transformed=views)

    def do_communication(self, obs):
        '''
        calculate message to send and transform memory based on received messages
        :param obs: agent made observation
        '''
        if self.init:
            self.init = False
            self.enemy_ids = [e.value for e in obs["enemies"]]
            self.teammate_pos = self.teammate_start[self.agent_id]
            self.prev_pos = obs["position"]
            self.memory.set_agent_spawns(obs["teammate"].value)

        if "message" in obs.keys():
            self.alter_memory(obs)
        self.calc_send_msg(obs)
        self.prev_pos = obs["position"]

    def alter_memory(self, obs):
        '''
        changes memory states for the agent based on the board and the received message
        :param obs: agent made observation including a message
        '''
        pos_change_received, enemy_received = _deserialize_msg(obs["message"])
        self.teammate_pos = tuple(
            map(lambda i, j: i + j, self.teammate_pos, self.pos_change_effect[pos_change_received]))
        # 10,11,12,13
        teammate_id = ((self.agent_id + 2) % 4) + 10
        self.memory.set_teammate_pos(obs['board'], teammate_id, self.teammate_pos)
        if enemy_received != 0:
            # this is always the opposite of sending bit due to one step message delay
            current_enemy_receive = self.enemy_ids[0] if not self.transmit_first else self.enemy_ids[1]
            self.memory.adjust_enemy_pos(obs['board'], current_enemy_receive, self.teammate_pos, enemy_received)

    def calc_send_msg(self, obs):
        '''
        calculate which message to send based on 2 types of information
        first information is an info of 5 possible values where the agent moved compare to last step
        second information is an info of 10 possible values in which region the agent sees the current enemy
        current enemy is always alternating between the two agents
        :param obs: agent made observation
        '''
        pos_change_message = self._get_position_change(obs["position"], self.prev_pos)
        view = crop_view(centralize_view(obs['board'], obs["position"], padding=0), 4)
        current_enemy_send = self.enemy_ids[0] if self.transmit_first else self.enemy_ids[1]

        enemy_message = 0 if current_enemy_send not in view else _region_of_enemy(view, current_enemy_send) + 1

        self.next_msg = _serialize_msg(pos_change_message, enemy_message)
        self.transmit_first = not self.transmit_first

    def act(self, obs, action_space):
        #filter before since it only works with the original observation
        valid_actions = get_filtered_actions(obs)

        if p.use_memory and p.p_observable:
        # only update memory here in initial state or in tournament mode
        # during training the data generator updates it pre-emptively with nobs
            if self.memory.is_none() or not self.is_train:
                self.update_memory(obs)
            act = self.policy(self.obs, valid_actions)
        else:
            act = self.policy(self.transformer(obs), valid_actions)
        act = int(act.detach().cpu().numpy()[0])
        if self.next_msg is not None:
            return [act, self.next_msg[0], self.next_msg[1]]
        else:
            return act

    def _get_position_change(self, t1, t2):
        '''
        helper to determine position change message
        :param t1: new postion tuple
        :param t2: old position tuple
        :return: an id according to the dict 'pos_change'
        '''
        sub = tuple(map(lambda i, j: i - j, t1, t2))
        return self.pos_change[sub]

    def reset_agent(self):
        '''
        reset agent to initial state (only needed in tournament)
        '''
        self.init = True
        self.prev_pos = None
        self.teammate_pos = None
        self.enemy_ids = None
        # bit signaling which agents position should be transmitted
        self.transmit_first = True
        self.memory.reset()

    def episode_end(self, reward):
        self.reset_agent()
