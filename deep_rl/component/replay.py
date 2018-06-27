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
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.__data = []
        self.__pos = 0

    def feed(self, experience):
        if self.__pos >= len(self.__data):
            self.__data.append(experience)
        else:
            self.__data[self.__pos] = experience
        self.__pos = (self.__pos + 1) % self.memory_size

    def feed_batch(self, experience):
        experience = zip(*experience)
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.__data)) for _ in range(batch_size)]
        sampled_data = [self.__data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data

    def size(self):
        return len(self.__data)

    def empty(self):
        return not len(self.__data)

class AsyncReplay(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    FEED_BATCH = 3
    def __init__(self, config):
        mp.Process.__init__(self)
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.__cache_len = 2
        self.start()

    def run(self):
        torch.cuda.is_available()
        replay = Replay(self.memory_size, self.batch_size)
        cache = []
        pending_batch = None

        first = True
        cur_cache = 0

        def set_up_cache():
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for i in range(self.__cache_len):
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
            op, data = self.__worker_pipe.recv()
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
                    self.__worker_pipe.send([cur_cache, cache])
                else:
                    self.__worker_pipe.send([cur_cache, None])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache)
                if pending_batch is not None:
                    for transition in pending_batch:
                        replay.feed(transition)
                    pending_batch = None
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def feed(self, exp):
        self.__pipe.send([self.FEED, exp])

    def feed_batch(self, exps):
        self.__pipe.send([self.FEED_BATCH, exps])

    def sample(self):
        self.__pipe.send([self.SAMPLE, None])
        cache_id, data = self.__pipe.recv()
        if data is not None:
            self.cache = data
        return self.cache[cache_id]

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

