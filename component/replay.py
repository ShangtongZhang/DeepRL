#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
import random
import torch.multiprocessing as mp

class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = None

        self.pos = 0
        self.full = False

    def feed(self, experience):
        if self.data is None:
            self.data = []
            for unit in experience:
                if np.isscalar(unit):
                    self.data.append(np.zeros(self.memory_size, dtype=type(unit)))
                else:
                    self.data.append(np.zeros((self.memory_size, ) + unit.shape, unit.dtype))
        for buffer_unit, exp_unit in zip(self.data, experience):
            buffer_unit[self.pos] = exp_unit

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def feed_batch(self, experience):
        experience = zip(*experience)
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=batch_size)
        return [unit[sampled_indices] for unit in self.data]

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def empty(self):
        return not self.full and not self.pos

class SkewedReplay:
    def __init__(self, memory_size, batch_size):
        memory_size = memory_size / 2
        self.non_zero_reward = Replay(memory_size, batch_size / 2)
        self.zero_reward = Replay(memory_size, batch_size / 2)
        self.batch_size = batch_size

    def feed(self, experiences):
        experiences = zip(*experiences)
        for exp in experiences:
            if np.abs(exp[2]) < 1e-5:
                self.zero_reward.feed(exp)
            else:
                self.non_zero_reward.feed(exp)

    def sample(self):
        if self.zero_reward.empty():
            batch = self.non_zero_reward.sample(self.batch_size)
        elif self.non_zero_reward.empty():
            batch = self.zero_reward.sample(self.batch_size)
        else:
            non_zero_batch_size = min(self.non_zero_reward.size(), self.batch_size / 2)
            zero_batch_size = min(self.zero_reward.size(), self.batch_size / 2)
            batch1 = self.zero_reward.sample(zero_batch_size)
            batch2 = self.non_zero_reward.sample(non_zero_batch_size)
            batch = list(map(lambda seq: np.concatenate([np.asarray(x) for x in seq], axis=0), zip(batch1, batch2)))
        batch = list(map(lambda x: np.asarray(x), batch))
        return batch




