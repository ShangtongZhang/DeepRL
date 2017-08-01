# Adapted from https://github.com/kvfrans/parallel-trpo/blob/master/utils.py
class Shifter:
    def __init__(self, filter_mean=True):
        self.m = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def state_dict(self):
        return {'m': self.m,
                'v': self.v,
                'n': self.n}

    def load_state_dict(self, saved):
        self.m = saved['m']
        self.v = saved['v']
        self.n = saved['n']

    def __call__(self, o):
        self.m = self.m * (self.n / (self.n + 1)) + o * 1 / (1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (o - self.m) ** 2 * 1 / (1 + self.n)
        self.std = (self.v + 1e-6) ** .5  # std
        self.n += 1
        if self.filter_mean:
            o_ = (o - self.m) / self.std
        else:
            o_ = o / self.std
        return o_
