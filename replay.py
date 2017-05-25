#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

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

    def sample(self, history_length):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []
        sampled_terminals = []
        for index in sampled_indices:
            if history_length == 1:
                sampled_states.append(self.states[index])
                sampled_next_states.append(self.next_states[index])
            else:
                full_indices = [(index - i + self.memory_size) % self.memory_size for i in range(history_length)]
                if self.pos in full_indices:
                    for i in range(full_indices.index(self.pos), len(full_indices)):
                        full_indices[i] = self.pos
                state = [self.states[i] for i in full_indices]
                state = np.vstack(state)
                sampled_states.append(state)

                next_state = [self.next_states[i] for i in full_indices]
                next_state = np.vstack(next_state)
                sampled_next_states.append(next_state)

            sampled_rewards.append(self.rewards[index])
            sampled_actions.append(self.actions[index])
            sampled_terminals.append(self.terminals[index])

        return [np.asarray(sampled_states),
                np.asarray(sampled_actions),
                np.asarray(sampled_rewards),
                np.asarray(sampled_next_states),
                np.asarray(sampled_terminals)]
