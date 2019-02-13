import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class State:
    def __init__(self, id, phi=None):
        self.id = id
        self.phi = phi


class Chain:
    def __init__(self):
        pass

    def reset(self):
        self.state = State(0)
        return self.state

    def step(self, action):
        sid = self.state.id
        r = 0
        if sid == 3:
            r = 10
        elif sid == 4:
            r = 1

        if sid == 0:
            if action == 0:
                self.state = State(1)
            elif action == 1:
                self.state = State(4)
            else:
                raise NotImplementedError
        elif sid == 3:
            self.state = State(7)
        elif sid == 10:
            self.state = State(0)
        else:
            self.state = State(sid + 1)
        return r, self.state


class TabularAgent:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.v = torch.zeros((11, ))
        self.c = torch.zeros((11, ))
        # self.pi = torch.randn((2, ), requires_grad=True)
        self.pi = torch.zeros((2, ), requires_grad=True)
        self.env = Chain()
        self.state = self.env.reset()

        self.a0 = 0
        self.s0 = 0
        self.a1 = 0
        self.s1 = 0

    def prob(self, state, action):
        if state.id:
            return torch.ones((1, ))
        prob = F.softmax(self.pi, dim=0)
        return prob[action]

    def act(self, state):
        if state.id:
            return 0
        if np.random.rand() < self.params['up_prob']:
            return 0
        else:
            return 1

    def rho(self, state, action):
        if state.id:
            return 1
        prob = self.prob(state, action)
        if action == 0:
            mu = self.params['up_prob']
        else:
            mu = 1 - self.params['up_prob']
        return prob / mu

    def value(self, state):
        return self.v[state.id]

    def learn_v(self, trajectory):
        with torch.no_grad():
            for s, a, r, next_s in trajectory:
                target = r + self.params['gamma'] * self.value(next_s)
                rho = self.rho(s, a)
                delta = target - self.value(s)
                self.v[s.id] = self.value(s) + self.params['v_lr'] * rho * delta

    def learn_c(self, trajectory):
        for s, a, r, next_s in trajectory:
            rho = self.rho(s, a)
            self.c[next_s.id] = self.c[next_s.id] + self.params['c_lr'] * (
                    self.params['gamma_hat'] * rho * self.c[s.id] +
                    (1 - self.params['gamma_hat']) - self.c[next_s.id])

    def generate_trajectory(self):
        trajectory = []
        for i in range(self.params['T']):
            action = self.act(self.state)
            r, next_s = self.env.step(action)
            trajectory.append([self.state, action, r, next_s])
            self.state = next_s
        return trajectory

    def run(self):
        trajectory = self.generate_trajectory()
        self.learn_v(trajectory)
        self.learn_c(trajectory)
        print(self.v)
        print(self.c)

if __name__ == '__main__':
    params = dict(
        up_prob=0.2,
        v_lr=0.01,
        c_lr=0.01,
        T=100000,
        gamma=0.1,
        gamma_hat=0.9,
    )
    agent = TabularAgent(**params)
    agent.run()
