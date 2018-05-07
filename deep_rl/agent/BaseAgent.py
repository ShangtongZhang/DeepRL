#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.evaluation_env = self.config.evaluation_env
        if self.evaluation_env is not None:
            self.evaluation_state = self.evaluation_env.reset()
            self.evaluation_return = 0

    def close(self):
        if hasattr(self.task, 'close'):
            self.task.close()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        action = self.network.predict(state, to_numpy=True)
        self.config.state_normalizer.unset_read_only()
        return np.argmax(action.flatten())

    def evaluate(self, steps=1):
        config = self.config
        if config.evaluation_env is None:
            return
        for _ in range(steps):
            action = self.evaluation_action(self.evaluation_state)
            self.evaluation_state, reward, done, _ = self.evaluation_env.step(action)
            self.evaluation_return += reward
            if done:
                self.evaluation_state = self.evaluation_env.reset()
                self.config.logger.info('evaluation episode return: %f' % (self.evaluation_return))
                self.evaluation_return = 0
