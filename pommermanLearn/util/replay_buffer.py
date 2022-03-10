from collections import deque
from operator import itemgetter
import random
import numpy as np

import params as p


class ReplayBuffer:
    def __init__(self, replay_size, n_step=1, gamma=p.gamma):
        self.replay_size = replay_size
        # prioritized replay variables
        if p.prioritized_replay:
            self.alpha = p.alpha
            self.priority_sums = [0 for _ in range(2 * self.replay_size)]  # store priorities in binary segment trees
            self.priority_mins = [float('inf') for _ in range(2 * self.replay_size)]
            self.max_priorities = 1.0
            self.sizes = 0

        self.buffers = []
        self.idxs = 0
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def add_to_buffer(self, obs, act, rwd, nobs, done):
        transition = (obs, act, rwd, nobs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        rwd_ins, nobs_ins, done_ins = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            rwd_ins = r + self.gamma * rwd_ins * (1 - d)
            nobs_ins, done_ins = (n_o, d) if d else (nobs_ins, done_ins)


        obs_ins, act_ins = self.n_step_buffer[0][:2]

        if len(self.buffers) < self.replay_size:
            self.buffers.append([obs_ins, act_ins, [rwd_ins], nobs_ins, [done_ins]])
        else:
            self.buffers[self.idxs] = [obs_ins, act_ins, [rwd_ins], nobs_ins, [done_ins]]

        if p.prioritized_replay:
            # set priority of new transitions to max_priority
            priority_alpha = self.max_priorities ** self.alpha

            self._set_priority_min(self.idxs, priority_alpha)
            self._set_priority_sum(self.idxs, priority_alpha)

            # increment size of replay buffer
            self.sizes = min(self.replay_size, self.sizes + 1)

        self.idxs = (self.idxs + 1) % self.replay_size
        return 1

    def _set_priority_min(self, idx, priority_alpha):
        ''' Update the minimum priotity tree'''
        # look at leaves of binary tree
        idx += self.replay_size
        self.priority_mins[idx] = priority_alpha
        # update whole tree
        while idx >= 2:
            idx //= 2  # index of parent
            self.priority_mins[idx] = min(self.priority_mins[2 * idx],
                                                     self.priority_mins[2 * idx + 1])

    def _set_priority_sum(self, idx, priority_alpha):
        ''' Update the maximum priority tree'''
        # look at leaves of tree
        idx += self.replay_size
        self.priority_sums[idx] = priority_alpha
        # update whole tree
        while idx >= 2:
            idx //= 2  # index of parent
            self.priority_sums[idx] = self.priority_sums[2 * idx] + self.priority_sums[
                2 * idx + 1]

    def _sum(self):
        ''' get sum of priorities'''
        return self.priority_sums[1]

    def _min(self):
        ''' get min of priorities'''
        return self.priority_mins[1]

    def find_prefix_sum_idx(self, prefix_sum):
        ''' find smallest index, s.t. the sum up to that index is greater or equal to prefix_sum'''
        idx = 1  # start from root
        while idx < self.replay_size:
            if self.priority_sums[
                idx * 2] > prefix_sum:  # if sum of left branch is bigger, go to left branch
                idx = 2 * idx
            else:  # else go to right branch and substract sum of left branch
                prefix_sum -= self.priority_sums[idx * 2]
                idx = 2 * idx + 1
        return idx - self.replay_size

    def get_batch_buffer(self, size, beta=p.beta):
        ''' sample transitions from buffer (including weights and indexes with prioritized replay '''
        if not p.prioritized_replay:
            indexed_samples = random.sample(list(enumerate(self.buffers)), size)
            indexes, samples = zip(*indexed_samples)
            weights = np.ones(shape=size, dtype=np.float32)
            batch = list(zip(*samples))
            return np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), weights, indexes
        else:
            samples = {
                'weights': np.zeros(shape=size, dtype=np.float32),
                'indexes': np.zeros(shape=size, dtype=np.int32)
            }

            # sample indexes according to probability
            for i in range(size):
                prefix_sum = random.random() * self._sum()
                idx = self.find_prefix_sum_idx(prefix_sum)
                samples['indexes'][i] = idx
            # calculate max weight (used to calculate individual weights)
            prob_min = self._min() / self._sum()
            max_weight = (prob_min * self.sizes) ** (-beta)

            # calculate weights
            for i in range(size):
                idx = samples['indexes'][i]
                prob = self.priority_sums[idx + self.replay_size] / self._sum()
                weight = (prob * self.sizes) ** (-beta)
                samples['weights'][i] = weight / max_weight

            # get sample transitions
            # transitions = list(zip(*np.array(self.buffers[agent_num])[samples['indexes']]))
            t = itemgetter(*samples['indexes'])(self.buffers)
            transitions = list(zip(*np.array(t)))
            return np.array(transitions[0]), np.array(transitions[1]), np.array(transitions[2]), np.array(
                transitions[3]), np.array(transitions[4]), \
                   samples['weights'], samples['indexes']

    def update_priorities(self, indexes, priorities):
        ''' update priorities of transitions'''
        for idx, priority in zip(indexes, priorities):
            priority = abs(priority.item()) + 1e-7
            self.max_priorities = max(self.max_priorities, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def get_batch_from_indices(self, idxs):
        samples = list(zip(*np.array(itemgetter(*idxs)(self.buffers))))
        return np.array(samples[0]), np.array(samples[1]), np.array(samples[2]), np.array(
            samples[3]), np.array(samples[4])