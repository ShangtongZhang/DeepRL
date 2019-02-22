import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_rl import get_logger
from deep_rl import mkdir
from deep_rl import set_one_thread
from deep_rl import set_tag
from deep_rl import Config
from deep_rl import random_seed
from deep_rl import Plotter
from deep_rl import translate
import pickle


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
        self.pi = nn.Parameter(torch.zeros((2,), requires_grad=True))
        self.opt = torch.optim.SGD([self.pi], lr=kwargs['pi_lr'])
        self.env = Chain()
        self.state = self.env.reset()

        self.v_check = Convergent(1e-3)
        self.c_check = Convergent(1e-5)
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
            return torch.zeros((1,), requires_grad=True)
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

    def cov_shift(self, state):
        return self.c[state.id]

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
            for i, (s, a, r, next_s) in enumerate(trajectory):
                rho = self.rho(s, a)
                self.c[next_s.id] = self.c[next_s.id] + self.params['c_lr'] * (
                        self.params['gamma_hat'] * rho * self.c[s.id] +
                        (1 - self.params['gamma_hat']) - self.c[next_s.id])
                if self.c_check(self.c):
                    return

    def emphatic_ac(self, trajectory):
        rho_prev = 0
        F = 0
        for i, (s, a, r, next_s) in enumerate(trajectory):
            self.learn_v([trajectory[i]])
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

            prob = self.prob(State(0))
            self.logger.add_scalar('p0', prob[0])
            # self.logger.add_scalar('p1', prob[1])
            # self.logger.add_scalar('v0', self.v[0])
            # self.logger.add_scalar('v1', self.v[1])
            # self.logger.add_scalar('v4', self.v[4])

    def compute_M1(self, trajectory):
        F = 0
        rho = 0
        for i, (s, a, r, next_s) in enumerate(trajectory):
            with torch.no_grad():
                c = self.cov_shift(s)
                F = self.params['gamma'] * rho * F + c
                M = (1 - self.params['lam_1']) * c + self.params['lam_1'] * F
                rho = self.rho(s, a)
        return M

    def compute_M2(self, trajectory):
        F = 0
        c_prev = 0
        rho_prev = 0
        grad_prev = torch.zeros((2,))
        for i, (s, a, r, next_s) in enumerate(trajectory):
            I = c_prev * rho_prev * grad_prev
            F = self.params['gamma_hat'] * rho_prev * F + I
            M = (1 - self.params['lam_2']) * I + self.params['lam_2'] * F

            rho_prev = self.rho(s, a)
            c_prev = self.cov_shift(s)
            if s.id:
                grad_prev = torch.zeros((2,))
            else:
                pi_loss = self.log_prob(s, a)
                self.opt.zero_grad()
                pi_loss.backward()
                grad_prev = self.pi.grad.clone()
        return M

    def generalized_ac(self, trajectory):
        F_1 = 0
        F_2 = 0
        rho_prev = 0
        c_prev = 0
        grad_log_pi_prev = torch.zeros((2, ))

        for i, (s, a, r, next_s) in enumerate(trajectory):
            self.learn_v([trajectory[i]])
            self.learn_c([trajectory[i]])
            with torch.no_grad():
                adv = r + self.params['gamma'] * self.value(next_s) - self.value(s)
                rho = self.rho(s, a)
                c = self.cov_shift(s)
                F_1 = self.params['gamma'] * rho_prev * F_1 + c
                M_1 = (1 - self.params['lam_1']) * c + self.params['lam_1'] * F_1
                I = c_prev * rho_prev * grad_log_pi_prev
                F_2 = self.params['gamma_hat'] * rho_prev * F_2 + I
                M_2 = (1 - self.params['lam_2']) * I + self.params['lam_2'] * F_2
                grad = self.params['gamma_hat'] * self.value(s) * M_2

                rho_prev = rho
                c_prev = c

            if s.id == 0:
                self.opt.zero_grad()
                self.log_prob(s, a).backward()
                grad_log_pi_prev = self.pi.grad.clone()
                self.pi.grad.mul_(rho * M_1 * adv).add_(grad).mul_(-1)
                self.opt.step()
            else:
                self.opt.zero_grad()
                grad_log_pi_prev = torch.zeros((2, ))
                self.pi._grad = -grad
                self.opt.step()

            prob = self.prob(State(0))
            self.logger.add_scalar('p0', prob[0])
            # self.logger.info('p0: %.2f' % (prob[0]))
            # self.logger.add_scalar('p1', prob[1])
            # self.logger.add_scalar('v0', self.v[0])
            # self.logger.add_scalar('v1', self.v[1])
            # self.logger.add_scalar('v4', self.v[4])
            # self.logger.add_scalar('c1', self.c[1])
            # self.logger.add_scalar('c4', self.c[4])

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
        if self.params['alg'] == 'ACE':
            self.emphatic_ac(trajectory)
        elif self.params['alg'] == 'GACE':
            self.generalized_ac(trajectory)

    def print(self):
        print('v0: %.2f' % (self.v[0]))
        print('v1: %.2f' % (self.v[1]))
        print('v4: %.2f' % (self.v[4]))
        print('d_mu_v1: %.2f' % (1 / 8 * self.params['up_prob'] * self.v[1]))
        print('d_mu_v4: %.2f' % (1 / 8 * (1 - self.params['up_prob']) * self.v[4]))


def tabular_agent(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('pi_lr', 0.01)
    kwargs.setdefault('v_lr', 0.1)
    kwargs.setdefault('c_lr', 0.1)
    kwargs.setdefault('alg', 'GACE')
    kwargs.setdefault('lam_2', 1)
    kwargs.setdefault('gamma_hat', 0.99)
    kwargs.setdefault('T', 10000)
    params = dict(
        up_prob=0.5,
        gamma=0.6,
        lam_1=1,
        logger=get_logger(tag=kwargs['tag'], skip=False)
    )
    params.update(kwargs)
    agent = TabularAgent(**params)
    agent.run()


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def batch():
    cf = Config()
    cf.add_argument('--i1', type=int, default=0)
    cf.add_argument('--i2', type=int, default=0)
    cf.merge()

    params = [
        dict(alg='ACE'),
    ]
    coef = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for lam2 in coef:
        for gamma_hat in coef:
            params.append(dict(alg='GACE', lam2=lam2, gamma_hat=gamma_hat))

    params = list(split(params, 5))
    for param in params[cf.i1]:
        tabular_agent(game='MDP', run=cf.i2, **param)

    exit()


if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    random_seed()
    # batch()

    # extract_heatmap_data()
    # tabular_agent(game='MDP', alg='GACE', gamma_hat=0.3)
