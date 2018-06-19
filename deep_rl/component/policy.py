#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

class GreedyPolicy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def sample(self, action_value, deterministic=False):
        random_action_prob = self.epsilon(0)
        if deterministic:
            return np.argmax(action_value)
        if np.random.rand() < random_action_prob:
            return np.random.randint(0, len(action_value))
        return np.argmax(action_value)

    def update_epsilon(self, steps=1):
        self.epsilon(steps)

class StochasticGreedyPolicy:
    def __init__(self, epsilons, final_step, min_epsilons, probs):
        self.policies = []
        self.probs = probs
        for epsilon, min_epsilon in zip(epsilons, min_epsilons):
            self.policies.append(GreedyPolicy(epsilon, final_step, min_epsilon))

    def sample(self, action_value, deterministic=False):
        return np.random.choice(self.policies, p=self.probs).sample(action_value, deterministic)

    def update_epsilon(self):
        for policy in self.policies:
            policy.update_epsilon()

class SamplePolicy:
    def sample(self, action_value, deterministic=False):
        if deterministic:
            return np.argmax(action_value)
        return np.random.choice(np.arange(len(action_value)), p=action_value)
    def update_epsilon(self):
        pass

class GaussianPolicy:
    def sample(self, mean, std, deterministic=False):
        if deterministic:
            return mean
        return mean + std * np.random.randn(*mean.shape)

    def update_epsilon(self):
        pass
