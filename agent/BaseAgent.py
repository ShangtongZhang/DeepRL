#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np

class BaseAgent:
    def __init__(self):
        self.testing = False

    def close(self):
        if hasattr(self.task, 'close'):
            self.task.close()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def deterministic_test(self):
        if self.testing:
            return
        if not self.config.test_interval:
            return
        if self.total_steps % self.config.test_interval:
            return
        if not hasattr(self, 'episode'):
            return
        rewards = []
        self.testing = True
        for _ in range(self.config.test_repetitions):
            rewards.append(self.episode(deterministic=True))
        self.testing = False
        self.config.logger.info('%d deterministic episodes: %f(%f)' % (
            self.config.test_repetitions, np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))
        ))
