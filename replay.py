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

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []


    def feed(self, experience):
        state, action, reward, next_state, done = experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(done)
        if len(self.terminals) > self.memory_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.terminals.pop(0)

    def sample(self):
        if len(self.terminals) >= self.batch_size:
            sampled_indices = np.arange(len(self.terminals))
            np.random.shuffle(sampled_indices)
            sampled_indices = sampled_indices[: self.batch_size]
            return [np.asarray(self.states)[sampled_indices],
                    np.asarray(self.actions)[sampled_indices],
                    np.asarray(self.rewards)[sampled_indices],
                    np.asarray(self.next_states)[sampled_indices],
                    np.asarray(self.terminals)[sampled_indices]]
        return None

