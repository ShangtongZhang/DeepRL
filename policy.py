import numpy as np

class GreedyPolicy:
    def __init__(self, epsilon, decay_factor, min_epsilon):
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.min_epsilon = min_epsilon

    def sample(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(state))
        return np.argmax(state)

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_factor
