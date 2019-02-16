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
                sub_trajectory = trajectory[max(0, i - self.params['window']): i + 1]
                self.learn_v(sub_trajectory)

                prob = self.prob(s)
                self.logger.add_scalar('p0', prob[0])
                self.logger.add_scalar('p1', prob[1])
                self.logger.add_scalar('v0', self.v[0])
                self.logger.add_scalar('v1', self.v[1])
                self.logger.add_scalar('v4', self.v[4])

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
        for i, (s, a, r, next_s) in enumerate(trajectory):
            if i < self.params['window']:
                continue
            with torch.no_grad():
                adv = r + self.params['gamma'] * self.value(next_s) - self.value(s)
                rho = self.rho(s, a)
            sub_trajectory = trajectory[max(i - self.params['window'], 0): i + 1]
            M_1 = self.compute_M1(sub_trajectory)
            M_2 = self.compute_M2(sub_trajectory)
            M_2 = self.params['gamma_hat'] * self.value(s) * M_2
            M_2 = M_2.detach()
            if s.id == 0:
                pi_loss = -rho * M_1 * adv * self.log_prob(s, a)
                self.opt.zero_grad()
                pi_loss.backward()
                self.pi.grad.add_(-M_2)
                self.opt.step()
            else:
                self.opt.zero_grad()
                self.pi._grad = -M_2
                self.opt.step()

            self.learn_v(sub_trajectory)
            self.learn_c(sub_trajectory)

            prob = self.prob(State(0))
            self.logger.add_scalar('p0', prob[0])
            self.logger.info('p0: %.2f' % (prob[0]))
            self.logger.add_scalar('p1', prob[1])
            self.logger.add_scalar('v0', self.v[0])
            self.logger.add_scalar('v1', self.v[1])
            self.logger.add_scalar('v4', self.v[4])
            self.logger.add_scalar('c1', self.c[1])
            self.logger.add_scalar('c4', self.c[4])

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
        if self.params['alg'] == 'ACE':
            self.emphatic_ac(trajectory)
        elif self.params['alg'] == 'GACE':
            self.learn_c(trajectory)
            self.generalized_ac(trajectory)
        # self.print()
        # print(self.v)
        # print(self.c)
        # print(F.softmax(self.pi, dim=0))

    def print(self):
        print('v0: %.2f' % (self.v[0]))
        print('v1: %.2f' % (self.v[1]))
        print('v4: %.2f' % (self.v[4]))
        print('d_mu_v1: %.2f' % (1 / 8 * self.params['up_prob'] * self.v[1]))
        print('d_mu_v4: %.2f' % (1 / 8 * (1 - self.params['up_prob']) * self.v[4]))


def read_tf_log(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    print(event_acc.Tags())
    w_times, step_nums, vals = zip(*event_acc.Scalars('p0'))


def tabular_agent(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('window', 10)
    kwargs.setdefault('pi_lr', 0.001)
    kwargs.setdefault('v_lr', 0.01)
    kwargs.setdefault('c_lr', 0.01)
    kwargs.setdefault('alg', 'GACE')
    kwargs.setdefault('lam_2', 1)
    kwargs.setdefault('gamma_hat', 0.99)
    params = dict(
        up_prob=0.5,
        T=100000,
        gamma=0.6,
        lam_1=1,
        logger=get_logger(tag=kwargs['tag'], skip=False)
    )
    params.update(kwargs)
    agent = TabularAgent(**params)
    agent.run()


def batch():
    cf = Config()
    cf.add_argument('--i1', type=int, default=0)
    cf.add_argument('--i2', type=int, default=0)
    cf.merge()

    params = [
        dict(window=10000),
        dict(window=1000),
        dict(window=100),
        dict(window=10),
        dict(window=10000, pi_lr=0.01),
        dict(window=10000, lam_2=0.9),
        dict(window=10000, lam_2=0.5),
        dict(window=10000, lam_2=0),
    ]

    tabular_agent(game='MDP', alg='GACE', run=cf.i2, **params[cf.i1])

    exit()


if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    random_seed()

    tabular_agent(game='MDP', alg='ACE')

    # batch()

    # read_tf_log('./tf_log/logger-MDP-190214-231348/events.out.tfevents.1550186028.c43b8419fa46')
    # read_tf_log('./tf_log/logger-MDP-190214-231348')

    # params.update(dict(pi_lr=0.001, alg='GACE'))
    # params.update(dict(pi_lr=0.01, alg='ACE'))

    # agent = TabularAgent(**params)
    # agent.run()