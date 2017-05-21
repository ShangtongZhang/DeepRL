#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size

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
            self.states = np.empty((self.memory_size, ) + state.shape)
            self.next_states = np.empty((self.memory_size, ) + state.shape)

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
        sampled_states = self.states[sampled_indices]
        sampled_actions = self.actions[sampled_indices]
        sampled_rewards = self.rewards[sampled_indices]
        sampled_next_states = self.next_states[sampled_indices]
        sampled_terminals = self.terminals[sampled_indices]
        return [sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminals]
