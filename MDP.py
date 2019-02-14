import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_rl import get_logger


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
            r = 5

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

class Convergent:
    def __init__(self, threshold, min_success=100):
        self.data = None
        self.threshold = threshold
        self.n_success = 0
        self.min_success = min_success

    def __call__(self, data):
        if self.data is None:
            self.data = data.clone()
            return False
        dist = (self.data - data).pow(2).mean().sqrt().item()
        self.data = data.clone()
        if dist < self.threshold:
            self.n_success += 1
        else:
            self.n_success = 0
        success = self.n_success > self.min_success
        if success:
            self.reset()
        return success

    def reset(self):
        self.data = None
        self.n_success = 0


class TabularAgent:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.v = torch.zeros((11,))
        self.c = torch.zeros((11,))
        # self.pi = torch.randn((2, ), requires_grad=True)
        self.pi = nn.Parameter(torch.zeros((2,), requires_grad=True))
        self.opt = torch.optim.SGD([self.pi], lr=params['pi_lr'])
        self.env = Chain()
        self.state = self.env.reset()

        self.v_check = Convergent(1e-3)
        self.logger = kwargs['logger']


    def prob(self, state, action=None):
        if state.id:
            return torch.ones((1,))
        prob = F.softmax(self.pi, dim=0)
        if action is None:
            return prob
        return prob[action]

    def log_prob(self, state, action):
        if state.id:
            return torch.zeros((1, ), requires_grad=True)
        log_prob = F.log_softmax(self.pi, dim=0)
        return log_prob[action]

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
            for i, (s, a, r, next_s) in enumerate(trajectory):
                target = r + self.params['gamma'] * self.value(next_s)
                rho = self.rho(s, a)
                delta = target - self.value(s)
                self.v[s.id] = self.value(s) + self.params['v_lr'] * rho * delta
                if self.v_check(self.v):
                    return

    def learn_c(self, trajectory):
        with torch.no_grad():
            for s, a, r, next_s in trajectory:
                rho = self.rho(s, a)
                self.c[next_s.id] = self.c[next_s.id] + self.params['c_lr'] * (
                        self.params['gamma_hat'] * rho * self.c[s.id] +
                        (1 - self.params['gamma_hat']) - self.c[next_s.id])

    def emphatic_ac(self, trajectory):
        rho_prev = 0
        F = 0
        for i, (s, a, r, next_s) in enumerate(trajectory):
            with torch.no_grad():
                F = self.params['gamma'] * rho_prev * F + 1
                M = (1 - self.params['lam_1']) + self.params['lam_1'] * F
                rho = self.rho(s, a)
                adv = r + self.params['gamma'] * self.value(next_s) - self.value(s)
                rho_prev = rho
            if s.id == 0:
                pi_loss = -rho * M * adv * self.log_prob(s, a)
                self.opt.zero_grad()
                pi_loss.backward()
                self.opt.step()
                self.learn_v(trajectory)

                prob = self.prob(s)
                self.logger.add_scalar('p0', prob[0])
                self.logger.add_scalar('p1', prob[1])
                self.logger.add_scalar('v0', self.v[0])
                self.logger.add_scalar('v1', self.v[1])
                self.logger.add_scalar('v4', self.v[4])

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
        # self.learn_c(trajectory)
        self.emphatic_ac(trajectory)
        self.print()
        # print(self.v)
        # print(self.c)
        # print(F.softmax(self.pi, dim=0))

    def print(self):
        print('v0: %.2f' % (self.v[0]))
        print('v1: %.2f' % (self.v[1]))
        print('v4: %.2f' % (self.v[4]))
        print('d_mu_v1: %.2f' % (1 / 8 * self.params['up_prob'] * self.v[1]))
        print('d_mu_v4: %.2f' % (1 / 8 * (1 - self.params['up_prob']) * self.v[4]))


if __name__ == '__main__':
    params = dict(
        up_prob=0.5,
        v_lr=0.01,
        c_lr=0.01,
        pi_lr=0.01,
        T=100000,
        gamma=0.6,
        gamma_hat=0.9,
        lam_1=1,
        logger=get_logger(tag='MDP', skip=False)
    )
    agent = TabularAgent(**params)
    agent.run()
