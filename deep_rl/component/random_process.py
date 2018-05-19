import numpy as np

class RandomProcess(object):
    def reset_states(self):
        pass

class GaussianProcess(RandomProcess):
    def __init__(self, size, std_schedules):
        self.size = size
        self.std_schedules = std_schedules

    def get_std(self):
        stds = [std() for std in self.std_schedules]
        return np.reshape(np.asarray(stds), (-1, 1))

    def sample(self):
        return np.random.randn(*self.size) * self.get_std()
