#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

class GreedyPolicy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def sample(self, action_value, deterministic=False):
        random_action_prob = self.epsilon(0)
        if deterministic:
            return np.argmax(action_value)
        if np.random.rand() < random_action_prob:
            return np.random.randint(0, len(action_value))
        return np.argmax(action_value)

    def update_epsilon(self, steps=1):
        self.epsilon(steps)

class SamplePolicy:
    def sample(self, action_value, deterministic=False):
        if deterministic:
            return np.argmax(action_value)
        return np.random.choice(np.arange(len(action_value)), p=action_value)
    def update_epsilon(self):
        pass
