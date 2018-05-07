import numpy as np

class RandomProcess(object):
    def reset_states(self):
        pass

class GaussianProcess(RandomProcess):
    def __init__(self, size, std_schedule):
        self.size = size
        self.std_schedule = std_schedule

    def sample(self):
        return np.random.randn(self.size) * self.std_schedule()
