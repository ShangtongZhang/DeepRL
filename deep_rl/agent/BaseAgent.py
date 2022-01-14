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
from skimage.io import imsave


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0
        self.average_reward_cache = []
        self.field = 'main'

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename,
                                map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        discounted_return = 0
        for r in reversed(info[0]['all_rewards']):
            discounted_return = r + self.config.discount * discounted_return
        return dict(ret=ret, dis_ret=discounted_return)

    def eval_episodes(self):
        episodic_returns = []
        discounted_returns = []
        for ep in range(self.config.eval_episodes):
            results = self.eval_episode()
            episodic_returns.append(results['ret'])
            discounted_returns.append(results['dis_ret'])
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f), discounted_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(
                episodic_returns) / np.sqrt(len(episodic_returns)),
            np.mean(discounted_returns), np.std(discounted_returns) /
            np.sqrt(len(discounted_returns))
        ))
        self.logger.add_scalar('%s_episodic_return_test' % (self.field), np.mean(
            episodic_returns), self.total_steps)
        self.logger.add_scalar('%s_discounted_return_test' % (self.field), np.mean(
            discounted_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                discounted_return = 0
                episode_len = len(info['all_rewards'])
                for r in reversed(info['all_rewards']):
                    discounted_return = r + self.config.discount * discounted_return
                self.logger.add_scalar(
                    'discounted_return_train', discounted_return, self.total_steps + offset)
                self.logger.add_scalar(
                    'episodic_return_train', ret, self.total_steps + offset)
                self.logger.add_scalar(
                    'episode_len', episode_len, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s, discounted_return_train %s, episode_len %d' % (
                    self.total_steps + offset, ret, discounted_return, episode_len))
            self.average_reward_cache.append(info['reward'])
            if len(self.average_reward_cache) == self.config.rollout_length:
                avg_reward = np.mean(self.average_reward_cache)
                self.average_reward_cache = []
                self.logger.add_scalar(
                    'avg_reward', avg_reward, self.total_steps + offset
                )
                self.logger.info('steps %d, avg reward %s' %
                                 (self.total_steps + offset, avg_reward))

        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)


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
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

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
