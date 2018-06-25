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
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        experience = zip(*experience)
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

class AsyncReplay(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    def __init__(self, memory_size, batch_size):
        mp.Process.__init__(self)
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.__cache_len = 2
        self.start()

    def run(self):
        torch.cuda.is_available()
        replay = Replay(self.memory_size, self.batch_size)
        cache = deque([], maxlen=self.__cache_len)

        def sample():
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            cache.append(batch_data)

        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.FEED:
                replay.feed(data)
            elif op == self.SAMPLE:
                if len(cache) == 0:
                    sample()
                self.__worker_pipe.send(cache.popleft())
                while len(cache) < self.__cache_len:
                    sample()
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def feed(self, exp):
        self.__pipe.send([self.FEED, exp])

    def sample(self):
        self.__pipe.send([self.SAMPLE, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

