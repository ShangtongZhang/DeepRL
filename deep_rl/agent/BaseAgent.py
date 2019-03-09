#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
from ..utils import *
import torch.multiprocessing as mp
from collections import deque
import sys


class BaseAgent:
    def __init__(self, config):
        self.config = config

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def eval_step(self, state):
        raise Exception('eval_step not implemented')

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        total_rewards = []
        while True:
            action = self.eval_step(state)
            state, reward, done, _ = env.step(action)
            total_rewards.append(reward[0])
            if done[0]:
                break
        return total_rewards

    def compute_values(self, rewards):
        config = self.config
        values = rewards.copy()
        for i in reversed(range(len(values) - 1)):
            values[i] = values[i] + config.discount * values[i + 1]
        return values

    def eval_episodes(self):
        episodic_rewards = []
        values = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_rewards.append(np.sum(total_rewards))
            values.extend(self.compute_values(total_rewards))
        self.config.logger.info('total_steps %d, episodic_return %.2f(%.2f), averaged_value %.2f' % (
            self.total_steps, np.mean(episodic_rewards), np.std(episodic_rewards) / np.sqrt(len(episodic_rewards)), np.mean(values)
        ))
        self.config.logger.add_scalar('episodic_return', np.mean(episodic_rewards), self.total_steps)
        self.config.logger.add_scalar('averaged_value', np.mean(values), self.total_steps)
        return {
            'episodic_return': np.mean(episodic_rewards),
            'averaged_value': np.mean(values)
        }


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transitions.append(self._transition())
        return transitions

    def run(self):
        self._set_up()
        torch.cuda.is_available()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise Exception('Unknown command')

    def _transition(self):
        raise Exception('Not implemented')

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])
