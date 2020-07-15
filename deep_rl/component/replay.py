#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from ..utils import *
import random


class Replay:
    def __init__(self, memory_size, batch_size, n_step=1, discount=1, history_length=1, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['state', 'action', 'reward', 'mask',
                       'v', 'q', 'pi', 'log_pi', 'entropy',
                       'advantage', 'return', 'q_a', 'log_pi_a',
                       'mean', 'next_state']
        self.keys = keys
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.discount = discount
        self.pos = 0
        self.history_length = history_length
        self._size = 0
        self.reset()

    def size(self):
        return self._size

    def full(self):
        return self._size == self.memory_size

    def feed(self, data):
        for k, vs in data.items():
            if k not in self.keys:
                raise RuntimeError('Undefined key')
            storage = getattr(self, k)
            pos = self.pos
            size = self.size()
            for v in vs:
                if pos >= len(storage):
                    storage.append(v)
                    size += 1
                else:
                    storage[self.pos] = v
                pos = (pos + 1) % self.memory_size
        self.pos = pos
        self._size = size

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.memory_size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])
        self.pos = 0
        self._size = 0

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_data = []
        while len(sampled_data) < batch_size:
            transition = self.retrive_transition(np.random.randint(0, self.size()))
            if transition is not None:
                sampled_data.append(transition)
        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def safe_index(self, i):
        return i % self.memory_size

    def retrive_transition(self, index):
        s_start = index - self.history_length + 1
        s_end = index
        if s_start < 0:
            return None
        next_s_start = s_start + self.n_step
        next_s_end = s_end + self.n_step
        if s_end < self.pos and next_s_end >= self.pos:
            return None

        state = [self.state[self.safe_index(i)] for i in range(s_start, s_end + 1)]
        next_state = [self.state[self.safe_index(i)] for i in range(next_s_start, next_s_end + 1)]
        action = self.action[s_end]
        reward = [self.reward[self.safe_index(i)] for i in range(s_end, s_end + self.n_step)]
        mask = [self.mask[self.safe_index(i)] for i in range(s_end, s_end + self.n_step)]
        state = np.asarray(state)
        next_state = np.asarray(next_state)
        cum_r = 0
        cum_mask = 1
        for i in reversed(np.arange(self.n_step)):
            cum_r = reward[i] + mask[i] * self.discount * cum_r
            cum_mask = cum_mask and mask[i]
        return [state, action, cum_r, next_state, cum_mask]


class PrioritizedReplay:
    def __init__(self, memory_size, batch_size):
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_priority = 1

    def feed(self, experience):
        self.tree.add(self.max_priority, experience)

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.asarray(priorities) / self.tree.total()

        sampled_data = []
        for i in range(batch_size):
            exp = []
            exp.extend(batch[i])
            exp.append(sampling_probabilities[i])
            exp.append(idxs[i])
            sampled_data.append(exp)

        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def update_priorities(self, info):
        for idx, priority in info:
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)


class AsyncReplay(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    UPDATE_PRIORITIES = 3

    def __init__(self, replay_params, replay_type=Config.DEFAULT_REPLAY):
        mp.Process.__init__(self)
        self.pipe, self.worker_pipe = mp.Pipe()
        self.replay_params = replay_params
        self.cache_len = 2
        self.replay_type = replay_type
        self.start()

    def run(self):
        if self.replay_type == Config.DEFAULT_REPLAY:
            replay = Replay(**self.replay_params)
        elif self.replay_type == Config.PRIORITIZED_REPLAY:
            replay = PrioritizedReplay(**self.replay_params)
        else:
            raise NotImplementedError

        cache = []
        pending_data = None

        first = True
        cur_cache = 0

        def set_up_cache():
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for i in range(self.cache_len):
                cache.append([x.clone() for x in batch_data])
                for x in cache[i]: x.share_memory_()
            sample(0)
            sample(1)

        def sample(cur_cache):
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for cache_x, x in zip(cache[cur_cache], batch_data):
                cache_x.copy_(x)

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.FEED:
                if not first:
                    pending_data = data
                else:
                    replay.feed(data)
            elif op == self.SAMPLE:
                if first:
                    set_up_cache()
                    first = False
                    self.worker_pipe.send([cur_cache, cache])
                else:
                    self.worker_pipe.send([cur_cache, None])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache)
                if pending_data is not None:
                    replay.feed(pending_data)
                    pending_data = None
            elif op == self.UPDATE_PRIORITIES:
                replay.update_priorities(data)
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def feed(self, exp):
        self.pipe.send([self.FEED, exp])

    def sample(self):
        self.pipe.send([self.SAMPLE, None])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.cache = data
        return self.cache[cache_id]

    def update_priorities(self, info):
        self.pipe.send([self.UPDATE_PRIORITIES, info])

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()


class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
