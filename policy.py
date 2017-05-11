#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

class GreedyPolicy:
    def __init__(self, epsilon, end_episode, min_epsilon):
        self.init_epsilon = self.epsilon = epsilon
        self.current_episode = 0
        self.min_epsilon = min_epsilon
        self.end_episode = end_episode

    def sample(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(state))
        return np.argmax(state)

    def update_epsilon(self):
        self.epsilon = self.init_epsilon - float(self.current_episode) / self.end_episode * (self.init_epsilon - self.min_epsilon)
        self.epsilon = max(self.epsilon, self.min_epsilon)
        self.current_episode += 1
