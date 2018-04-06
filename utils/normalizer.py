#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch
import numpy as np

class RunningStatsNormalizer:
    def __init__(self):
        self.needs_reset = True

    def reset(self, x_size):
        self.m = np.zeros(x_size)
        self.v = np.zeros(x_size)
        self.n = 0.0
        self.needs_reset = False

    def __call__(self, x):
        if np.isscalar(x) or len(x.shape) == 1:
            if self.needs_reset: self.reset(1)
            return self.nomalize_single(x)
        elif len(x.shape) == 2:
            if self.needs_reset: self.reset(x.shape[1])
            new_x = np.zeros(x.shape)
            for i in range(x.shape[0]):
                new_x[i] = self.nomalize_single(x[i])
            return new_x
        else:
            assert 'Unsupported Shape'

    def nomalize_single(self, x):
        is_scalar = np.isscalar(x)
        if is_scalar:
            x = np.asarray([x])
        new_m = self.m * (self.n / (self.n + 1)) + x / (self.n + 1)
        self.v = self.v * (self.n / (self.n + 1)) + (x - self.m) * (x - new_m) / (self.n + 1)
        self.m = new_m
        self.n += 1

        std = (self.v + 1e-6) ** .5
        x = (x - self.m) / std
        if is_scalar:
            x = np.asscalar(x)
        return x

class RescaleNormalizer:
    def __init__(self, coef=1.0):
        self.coef = coef

    def __call__(self, x):
        return self.coef * x

class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0 / 255)

class SignNormalizer:
    def __call__(self, x):
        return np.sign(x)