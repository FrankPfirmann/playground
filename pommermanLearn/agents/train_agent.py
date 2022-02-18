import params
from action_prune import get_filtered_actions
from pommerman import agents
from pommerman.constants import Action
from util.data import centralize_view, crop_view, merge_views, merge_views_life
import numpy as np
import params as p


class TrainAgent(agents.BaseAgent):
    """
    An agent used for training a pytorch model 
    """

    def __init__(self, policy, transformer, communicate=params.communicate):
        """
        Initializes the agent and sets the model to train 

        :param policy: The pytorch model to train
        """
        super(TrainAgent, self).__init__()
        self.policy = policy

        self.init = True
        self.prev_pos = None
        self.teammate_pos = None
        self.enemy_ids = None
        # bit signaling which agents position should be transmitted
        self.transmit_first = True
        self.memory = None
        self.pos_change = {(0, 0): Action.Stop.value,
                           (-1, 0): Action.Up.value,
                           (1, 0): Action.Down.value,
                           (0, 1): Action.Right.value,
                           (0, -1): Action.Left.value}

        self.pos_change_effect = {v: k for k, v in self.pos_change.items()}
        self.teammate_start = {0: (9, 9),
                               1: (1, 9),
                               2: (0, 0),
                               3: (9, 1)}
        self.communicate = communicate
        self.transformer = transformer

    def update_memory(self, nobs, set_memory=True):
        if self.memory is None or not self.validate_memory(nobs):
            self.memory = nobs
            return

        # Invert fog to get field of view
        fov = 1 - nobs[0][..., 14, :, :]

        for layer in range(14):
            if layer in [0, 1]:  # Remember walls and passages always
                forgetfulness = 0.0
            else:  # Forget other layers that are out of view slowly
                forgetfulness = p.forgetfullness
            first = self.memory[0][..., layer, :, :]
            second = nobs[0][..., layer, :, :]
            if layer <= 12:
                merged = merge_views(first, second, fov, forgetfullness=forgetfulness)
            else:
                merged = merge_views_life(first, second, fov)
            nobs[0][..., layer, :, :] = merged
        if set_memory:
            self.memory = nobs
        else:
            return nobs

    def get_memory(self):
        return self.memory

    def get_memory_view(self, nobs):
        return self.update_memory(self.transformer(nobs), set_memory=False)

    def act(self, obs, action_space):
        # Initialize values at first time step and send no message
        if not self.communicate:
            #filter before since it only works with the original observation
            valid_actions = get_filtered_actions(obs)
            obs = self.transformer(obs)
            if p.use_memory and p.p_observable:
                # Memory in this form only makes sense with partial
                # observability
                self.update_memory(obs)
                obs = self.memory
            act = self.policy(obs, valid_actions)
            return int(act.detach().cpu().numpy()[0])
        if self.init:
            self.init = False
            self.enemy_ids = [e.value for e in obs["enemies"]]
            self.teammate_pos = self.teammate_start[self.agent_id]
            act = self.policy(obs)
            self.prev_pos = self.position
            return int(act.detach().cpu().numpy()[0])
        else:
            pos_change_message = self._get_position_change(self.position, self.prev_pos)
            act = self.policy(obs)
            if "message" in obs.keys():
                pos_change_received, enemy_received = self._deserialize_msg(obs["message"])
                self.teammate_pos = tuple(map(lambda i, j: i + j, self.teammate_pos, self.pos_change_effect[pos_change_received]))
            view = crop_view(centralize_view(obs['board'], self.position, padding=0), 4)
            current_enemy = self.enemy_ids[0] if self.transmit_first else self.enemy_ids[1]
            enemy_message = 0 if current_enemy not in view else self._region_of_enemy(view, current_enemy) + 1
            print(self.teammate_pos)
            print("Agent id " + str(self.agent_id) + " at pos" + str(self.position))
            msg1, msg2 = self._serialize_msg(pos_change_message, enemy_message)
            self.prev_pos = self.position
            return [int(act.detach().cpu().numpy()[0]), msg1, msg2]

    def _get_position_change(self, t1, t2):
        sub = tuple(map(lambda i, j: i - j, t1, t2))
        return self.pos_change[sub]

    def _region_of_enemy(self, view, id):
        index = np.where(view == id)
        return np.floor(index[0][0]/3)*3 + np.floor(index[1][0]/3)

    def _serialize_msg(self, pos, enemy):
        message = pos * 10 + enemy
        message2 = message % 8
        message1 = np.floor(message / 8)
        return int(message1), int(message2)

    def _deserialize_msg(self, msg):
        msg_total = msg[0] * 8 + msg[1]
        pos_change = np.floor(msg_total/10)
        enemy_msg = msg_total % 10
        return pos_change, enemy_msg

    def validate_memory(self, nobs):
        if self.memory is None:
            return False

        if self.memory[0].shape != nobs[0].shape:
            return False

        if self.memory[1].shape != nobs[1].shape:
            return False

        return True