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
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'mask'])
PrioritizedTransition = namedtuple('Transition',
                                   ['state', 'action', 'reward', 'next_state', 'mask', 'sampling_prob', 'idx'])


class Storage:
    def __init__(self, memory_size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['state', 'action', 'reward', 'mask',
                       'v', 'q', 'pi', 'log_pi', 'entropy',
                       'advantage', 'ret', 'q_a', 'log_pi_a',
                       'mean', 'next_state']
        self.keys = keys
        self.memory_size = memory_size
        self.reset()

    def feed(self, data):
        for k, v in data.items():
            if k not in self.keys:
                raise RuntimeError('Undefined key')
            getattr(self, k).append(v)

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

    def extract(self, keys):
        data = [getattr(self, k)[:self.memory_size] for k in keys]
        data = map(lambda x: torch.cat(x, dim=0), data)
        Entry = namedtuple('Entry', keys)
        return Entry(*list(data))


class UniformReplay(Storage):
    TransitionCLS = Transition

    def __init__(self, memory_size, batch_size, n_step=1, discount=1, history_length=1, keys=None):
        super(UniformReplay, self).__init__(memory_size, keys)
        self.batch_size = batch_size
        self.n_step = n_step
        self.discount = discount
        self.history_length = history_length
        self.pos = 0
        self._size = 0

    def compute_valid_indices(self):
        indices = []
        indices.extend(list(range(self.history_length - 1, self.pos - self.n_step)))
        indices.extend(list(range(self.pos + self.history_length - 1, self.size() - self.n_step)))
        return np.asarray(indices)

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

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_data = []
        while len(sampled_data) < batch_size:
            transition = self.construct_transition(np.random.randint(0, self.size()))
            if transition is not None:
                sampled_data.append(transition)
        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return Transition(*sampled_data)

    def valid_index(self, index):
        if index - self.history_length + 1 >= 0 and index + self.n_step < self.pos:
            return True
        if index - self.history_length + 1 >= self.pos and index + self.n_step < self.size():
            return True
        return False

    def construct_transition(self, index):
        if not self.valid_index(index):
            return None
        s_start = index - self.history_length + 1
        s_end = index
        if s_start < 0:
            raise RuntimeError('Invalid index')
        next_s_start = s_start + self.n_step
        next_s_end = s_end + self.n_step
        if s_end < self.pos and next_s_end >= self.pos:
            raise RuntimeError('Invalid index')

        state = [self.state[i] for i in range(s_start, s_end + 1)]
        next_state = [self.state[i] for i in range(next_s_start, next_s_end + 1)]
        action = self.action[s_end]
        reward = [self.reward[i] for i in range(s_end, s_end + self.n_step)]
        mask = [self.mask[i] for i in range(s_end, s_end + self.n_step)]
        if self.history_length == 1:
            # eliminate the extra dimension if no frame stack
            state = state[0]
            next_state = next_state[0]
        state = np.array(state)
        next_state = np.array(next_state)
        cum_r = 0
        cum_mask = 1
        for i in reversed(np.arange(self.n_step)):
            cum_r = reward[i] + mask[i] * self.discount * cum_r
            cum_mask = cum_mask and mask[i]
        return Transition(state=state, action=action, reward=cum_r, next_state=next_state, mask=cum_mask)

    def size(self):
        return self._size

    def full(self):
        return self._size == self.memory_size

    def update_priorities(self, info):
        raise NotImplementedError


class PrioritizedReplay(UniformReplay):
    TransitionCLS = PrioritizedTransition

    def __init__(self, memory_size, batch_size, n_step=1, discount=1, history_length=1, keys=None):
        super(PrioritizedReplay, self).__init__(memory_size, batch_size, n_step, discount, history_length, keys)
        self.tree = SumTree(memory_size)
        self.max_priority = 1

    def feed(self, data):
        super().feed(data)
        self.tree.add(self.max_priority, None)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        segment = self.tree.total() / batch_size

        sampled_data = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data_index) = self.tree.get(s)
            transition = super().construct_transition(data_index)
            if transition is None:
                continue
            sampled_data.append(PrioritizedTransition(
                *transition,
                sampling_prob=p / self.tree.total(),
                idx=idx,
            ))
        while len(sampled_data) < batch_size:
            # This should rarely happen
            sampled_data.append(random.choice(sampled_data))

        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        sampled_data = PrioritizedTransition(*sampled_data)
        return sampled_data

    def update_priorities(self, info):
        for idx, priority in info:
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)


class ReplayWrapper(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    UPDATE_PRIORITIES = 3

    def __init__(self, replay_cls, replay_kwargs, async=True):
        mp.Process.__init__(self)
        self.replay_kwargs = replay_kwargs
        self.replay_cls = replay_cls
        self.cache_len = 2
        if async:
            self.pipe, self.worker_pipe = mp.Pipe()
            self.start()
        else:
            self.replay = replay_cls(**replay_kwargs)
            self.sample = self.replay.sample
            self.feed = self.replay.feed
            self.update_priorities = self.replay.update_priorities

    def run(self):
        replay = self.replay_cls(**self.replay_kwargs)

        cache = []

        cache_initialized = False
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
                replay.feed(data)
            elif op == self.SAMPLE:
                if cache_initialized:
                    self.worker_pipe.send([cur_cache, None])
                else:
                    set_up_cache()
                    cache_initialized = True
                    self.worker_pipe.send([cur_cache, cache])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache)
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
        return self.replay_cls.TransitionCLS(*self.cache[cache_id])

    def update_priorities(self, info):
        self.pipe.send([self.UPDATE_PRIORITIES, info])

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()
