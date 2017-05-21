#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

class GreedyPolicy:
    def __init__(self, epsilon, final_step, min_epsilon):
        self.init_epsilon = self.epsilon = epsilon
        self.current_steps = 0
        self.min_epsilon = min_epsilon
        self.final_step = final_step

    def sample(self, action_value):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(action_value))
        return np.argmax(action_value)

    def update_epsilon(self):
        self.epsilon = self.init_epsilon - float(self.current_steps) / self.final_step * (self.init_epsilon - self.min_epsilon)
        self.epsilon = max(self.epsilon, self.min_epsilon)
        self.current_steps += 1

class SamplePolicy:
    def sample(self, action_value):
        return np.random.choice(np.arange(len(action_value)), p=action_value)
    def update_epsilon(self):
        pass