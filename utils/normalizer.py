#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch
import numpy as np

class Normalizer:
    def __init__(self, o_size):
        self.stats = SharedStats(o_size)

    def __call__(self, o_):
        if np.isscalar(o_):
            o = torch.FloatTensor([o_])
        else:
            o = torch.FloatTensor(o_)
        self.stats.feed(o)
        std = (self.stats.v + 1e-6) ** .5
        o = (o - self.stats.m) / std
        o = o.numpy()
        if np.isscalar(o_):
            o = np.asscalar(o)
        else:
            o = o.reshape(o_.shape)
        return o
   
    def state_dict(self):
        return self.stats.state_dict()

    def load_state_dict(self, saved):
        self.stats.load_state_dict(saved)

class StaticNormalizer:
    def __init__(self, o_size):
        self.offline_stats = SharedStats(o_size)
        self.online_stats = SharedStats(o_size)

    def __call__(self, o_):
        if np.isscalar(o_):
            o = torch.FloatTensor([o_])
        else:
            o = torch.FloatTensor(o_)
        self.online_stats.feed(o)
        if self.offline_stats.n[0] == 0:
            return o_
        std = (self.offline_stats.v + 1e-6) ** .5
        o = (o - self.offline_stats.m) / std
        o = o.numpy()
        if np.isscalar(o_):
            o = np.asscalar(o)
        else:
            o = o.reshape(o_.shape)
        return o
    
    def state_dict(self):
        return self.offline_stats.state_dict()

    def load_state_dict(self, saved):
        self.offline_stats.load_state_dict(saved)

class SharedStats:
    def __init__(self, o_size):
        self.m = torch.zeros(o_size)
        self.v = torch.zeros(o_size)
        self.n = torch.zeros(1)
        self.m.share_memory_()
        self.v.share_memory_()
        self.n.share_memory_()

    def feed(self, o):
        n = self.n[0]
        new_m = self.m * (n / (n + 1)) + o / (n + 1)
        self.v.copy_(self.v * (n / (n + 1)) + (o - self.m) * (o - new_m) / (n + 1))
        self.m.copy_(new_m)
        self.n.add_(1)

    def zero(self):
        self.m.zero_()
        self.v.zero_()
        self.n.zero_()

    def load(self, stats):
        self.m.copy_(stats.m)
        self.v.copy_(stats.v)
        self.n.copy_(stats.n)

    def merge(self, B):
        A = self
        n_A = self.n[0]
        n_B = B.n[0]
        n = n_A + n_B
        delta = B.m - A.m
        m = A.m + delta * n_B / n
        v = A.v * n_A + B.v * n_B + delta * delta * n_A * n_B / n
        v /= n
        self.m.copy_(m)
        self.v.copy_(v)
        self.n.add_(B.n)

    def state_dict(self):
        return {'m': self.m.numpy(),
                'v': self.v.numpy(),
                'n': self.n.numpy()}

    def load_state_dict(self, saved):
        self.m = torch.FloatTensor(saved['m'])
        self.v = torch.FloatTensor(saved['v'])
        self.n = torch.FloatTensor(saved['n'])
