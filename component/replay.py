#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
import random

class Replay:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype

        self.states = None
        self.actions = np.empty(self.memory_size, dtype=np.int8)
        self.rewards = np.empty(self.memory_size)
        self.next_states = None
        self.terminals = np.empty(self.memory_size, dtype=np.int8)

        self.pos = 0
        self.full = False


    def feed(self, experience):
        state, action, reward, next_state, done = experience

        if self.states is None:
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos][:] = next_state
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]

class HybridRewardReplay:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype

        self.states = None
        self.actions = np.empty(self.memory_size, dtype=np.int8)
        self.rewards = None
        self.next_states = None
        self.terminals = np.empty(self.memory_size, dtype=np.int8)

        self.pos = 0
        self.full = False


    def feed(self, experience):
        state, action, reward, next_state, done = experience

        if self.states is None:
            self.rewards = np.empty((self.memory_size, ) + reward.shape, dtype=self.dtype)
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos] = action
        self.rewards[self.pos][:] = reward
        self.next_states[self.pos][:] = next_state
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]

class HighDimActionReplay:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype

        self.states = None
        self.actions = None
        self.rewards = np.empty(self.memory_size)
        self.next_states = None
        self.terminals = np.empty(self.memory_size, dtype=np.int8)

        self.pos = 0
        self.full = False


    def feed(self, experience):
        state, action, reward, next_state, done = experience

        if self.states is None:
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.actions = np.empty((self.memory_size, ) + action.shape)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos][:] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos][:] = next_state
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]

class GeneralReplay:
    def __init__(self, memory_size, batch_size):
        self.buffer = []
        self.memory_size = memory_size
        self.batch_size = batch_size

    def feed(self, experiences):
        for experience in zip(*experiences):
            self.buffer.append(experience)
            if len(self.buffer) > self.memory_size:
                del self.buffer[0]

    def sample(self):
        sampled = zip(*random.sample(self.buffer, self.batch_size))
        return sampled

    def clear(self):
        self.buffer = []

    def full(self):
        return len(self.buffer) == self.memory_size
