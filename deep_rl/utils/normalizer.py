#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np

class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class RunningStatsNormalizer(BaseNormalizer):
    def __init__(self, read_only=False):
        BaseNormalizer.__init__(self, read_only)
        self.needs_reset = True
        self.read_only = read_only

    def reset(self, x_size):
        self.m = np.zeros(x_size)
        self.v = np.zeros(x_size)
        self.n = 0.0
        self.needs_reset = False

    def state_dict(self):
        return {'m': self.m, 'v': self.v, 'n': self.n}

    def load_state_dict(self, stored):
        self.m = stored['m']
        self.v = stored['v']
        self.n = stored['n']
        self.needs_reset = False

    def __call__(self, x):
        if np.isscalar(x) or len(x.shape) == 1:
            # if dim of x is 1, it can be interpreted as 1 vector entry or batches of scalar entry,
            # fortunately resetting the size to 1 applies to both cases
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

        if not self.read_only:
            new_m = self.m * (self.n / (self.n + 1)) + x / (self.n + 1)
            self.v = self.v * (self.n / (self.n + 1)) + (x - self.m) * (x - new_m) / (self.n + 1)
            self.m = new_m
            self.n += 1

        std = (self.v + 1e-6) ** .5
        x = (x - self.m) / std
        if is_scalar:
            x = np.asscalar(x)
        return x

class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        if not np.isscalar(x):
            x = np.asarray(x)
        return self.coef * x

class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0 / 255)

class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)