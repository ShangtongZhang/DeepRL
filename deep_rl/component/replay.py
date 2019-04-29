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


class Replay:
    def __init__(self, memory_size, batch_size, drop_prob=0, to_np=True):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0
        self.drop_prob = drop_prob
        self.to_np = to_np

    def feed(self, experience):
        if np.random.rand() < self.drop_prob:
            return
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if self.empty():
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        sampled_data = zip(*sampled_data)
        if self.to_np:
            sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def shuffle(self):
        np.random.shuffle(self.data)

    def clear(self):
        self.data = []
        self.pos = 0


class SkewedReplay:
    def __init__(self, memory_size, batch_size, criterion):
        self.replay1 = Replay(memory_size // 2, batch_size // 2)
        self.replay2 = Replay(memory_size // 2, batch_size // 2)
        self.criterion = criterion

    def feed(self, experience):
        if self.criterion(experience):
            self.replay1.feed(experience)
        else:
            self.replay2.feed(experience)

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self):
        data1 = self.replay1.sample()
        data2 = self.replay2.sample()
        if data2 is not None:
            data = list(map(lambda x: np.concatenate(x, axis=0), zip(data1, data2)))
        else:
            data = data1
        return data


class AsyncReplay(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    FEED_BATCH = 3

    def __init__(self, memory_size, batch_size):
        mp.Process.__init__(self)
        self.pipe, self.worker_pipe = mp.Pipe()
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.cache_len = 2
        self.start()

    def run(self):
        replay = Replay(self.memory_size, self.batch_size)
        cache = []
        pending_batch = None

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
                replay.feed(data)
            elif op == self.FEED_BATCH:
                if not first:
                    pending_batch = data
                else:
                    for transition in data:
                        replay.feed(transition)
            elif op == self.SAMPLE:
                if first:
                    set_up_cache()
                    first = False
                    self.worker_pipe.send([cur_cache, cache])
                else:
                    self.worker_pipe.send([cur_cache, None])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache)
                if pending_batch is not None:
                    for transition in pending_batch:
                        replay.feed(transition)
                    pending_batch = None
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def feed(self, exp):
        self.pipe.send([self.FEED, exp])

    def feed_batch(self, exps):
        self.pipe.send([self.FEED_BATCH, exps])

    def sample(self):
        self.pipe.send([self.SAMPLE, None])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.cache = data
        return self.cache[cache_id]

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
