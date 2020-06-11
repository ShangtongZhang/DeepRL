#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib.pyplot as plt


def plot_mean_std(data, x=None, **kwargs):
    if x is None:
        x = np.arange(data.shape[1])
    # e_x = np.std(data, axis=0) / np.sqrt(data.shape[0])
    e_x = np.std(data, axis=0)
    m_x = np.mean(data, axis=0)
    plt.plot(x, m_x, **kwargs)
    del kwargs['label']
    plt.fill_between(x, m_x + e_x, m_x - e_x, alpha=0.3, **kwargs)


DISCOUNT = 0.99
FEATURE_SIZE = 8
SOLID = 0
DASHED = 1
ACTIONS = [SOLID, DASHED]

LOWER_STATE = 6

P_Pi = np.zeros((7, 7))
P_Pi[:, -1] = 1
d_mu = np.ones((7, 1)) / 7.0
D_mu = np.diag(d_mu.flatten())
m_oracle = np.linalg.inv(D_mu) @ np.linalg.inv(np.eye(7) - DISCOUNT * P_Pi.T) @ d_mu


def get_state(phi):
    phi = phi[:-1]
    return np.argmax(phi)


class Baird:
    def __init__(self):
        self.states = np.arange(7)
        self.phi = np.zeros((7, 8))
        self.actions = [0, 1]
        for i in range(6):
            self.phi[i, i] = 2
            self.phi[i, -1] = 1
        self.phi[6, 6] = 1
        self.phi[6, 7] = 2

    def reset(self):
        self.state = np.random.choice(self.states)
        return self.phi[self.state]

    def step(self, action):
        if action == SOLID:
            self.state = 6
        elif action == DASHED:
            self.state = np.random.randint(6)
        else:
            raise NotImplementedError
        return self.phi[self.state], 0, False, {}


def mu(state):
    if np.random.rand() < 1.0 / 7:
        action = SOLID
    else:
        action = DASHED
    return action


def compute_rho(state, action):
    if action == 0:
        rho = 7
    elif action == 1:
        rho = 0
    else:
        raise NotImplementedError
    return rho


class ProximalETD:
    def __init__(self,
                 env,
                 lr1,
                 lr2,
                 beta=DISCOUNT,
                 use_oracle_v=False):
        self.env = env
        self.lr1 = lr1
        self.lr2 = lr2
        self.beta = beta
        self.use_oracle_v = use_oracle_v

        phi = self.env.phi
        A = phi.T @ (np.eye(7) - self.beta * P_Pi.T) @ D_mu @ phi
        C = phi.T @ D_mu @ phi
        b = phi.T @ d_mu
        G = np.block([[-C, -A], [A.T, np.zeros((8, 8))]])
        g = np.block([[b], [np.zeros((8, 1))]])
        self.w_oracle = -np.linalg.inv(G) @ g
        self.w_oracle = self.w_oracle.flatten()[8:]

    def reset(self):
        self.s = self.env.reset()
        # self.v = np.zeros(8)
        # self.w = np.zeros(8)
        self.v = np.random.randn(8)
        # self.w = np.random.randn(8) * 0.1
        self.w = self.w_oracle + np.random.randn(8) * 0.1
        self.theta = np.asarray([1.0, 1, 1, 1, 1, 1, 10, 1])

        self.rho = 0

        self.m_stats = {}
        for i in range(7):
            self.m_stats[i] = []

        self.n = 0

        self.v_targets = []

    def step(self):
        self.n += 1
        phi = self.env.phi

        # m_hat = phi @ self.w.reshape((8, 1))
        m_hat = phi @ self.w_oracle.reshape((8, 1))
        delta_w_bar = np.ones((7, 1)) + self.beta * np.linalg.inv(D_mu) @ P_Pi.T @ D_mu @ m_hat - m_hat
        term = phi.T @ D_mu @ delta_w_bar
        v = np.linalg.inv(phi.T @ D_mu @ phi) @ term
        J = term.T @ v
        print(J)

        # def sample_prob(s_bar, a_bar, s):
        #     if a_bar == SOLID:
        #         prob = 1.0 / 7 * (1 if s == LOWER_STATE else 0)
        #     elif a_bar == DASHED:
        #         prob = 6.0 / 7 * (0 if s == LOWER_STATE else 1.0 / 6)
        #     else:
        #         raise NotImplementedError
        #     return prob
        #
        # delta_w = 0
        # for s in range(7):
        #     target = 0
        #     for s_bar in range(7):
        #         for a_bar in ACTIONS:
        #             target += sample_prob(s_bar, a_bar, s) * compute_rho(s_bar, a_bar) * self.env.phi[s_bar]
        #     delta_w += (self.env.phi[s] - self.beta * target) * (self.env.phi[s] * v).sum()

        grad = phi.T @ D_mu @ (self.beta * np.linalg.inv(D_mu) @ P_Pi.T @ D_mu @ phi - phi)
        delta_w = grad.T @ v
        delta_w = -delta_w.flatten()

        self.w += self.lr1 / self.n * delta_w
        # self.w += self.lr1 * delta_w

        s = np.random.randint(7)
        a = mu(s)
        next_s = LOWER_STATE if a == SOLID else np.random.randint(LOWER_STATE)
        rho = compute_rho(s, a)

        delta = DISCOUNT * (phi[next_s] * self.theta).sum() - (phi[s] * self.theta).sum()
        self.theta += self.lr2 * (phi[s] * self.w).sum() * rho * delta * phi[s]

        self.m_stats[6].append((self.env.phi[6] * self.w).sum())

    def get_info(self):
        return np.copy(self.theta), self.m_stats, np.copy(self.w)


class ETD:
    def __init__(self,
                 env,
                 lr,
                 use_oracle_m=False):
        self.env = env
        self.lr = lr
        self.use_oracle_m = use_oracle_m

    def reset(self):
        self.s = self.env.reset()
        self.w = np.asarray([1.0, 1, 1, 1, 1, 1, 10, 1])
        self.M = 0
        self.rho = 0
        self.m_stats = {}
        for i in range(7):
            self.m_stats[i] = []

    def step(self):
        a = mu(self.s)
        next_s, r, done, _ = self.env.step(a)
        rho = compute_rho(self.s, a)
        delta = r + DISCOUNT * (next_s * self.w).sum() - (self.s * self.w).sum()
        M = DISCOUNT * self.M * self.rho + 1
        self.m_stats[get_state(self.s)].append(M)

        if self.use_oracle_m:
            M = m_oracle[get_state(self.s)]
        self.w += self.lr * M * rho * delta * self.s

        self.rho = rho
        self.s = next_s
        self.M = M

    def get_info(self):
        return np.copy(self.w), self.m_stats


class TD:
    def __init__(self, env, lr):
        self.env = env
        self.lr = lr

    def reset(self):
        self.s = self.env.reset()
        self.w = np.asarray([1.0, 1, 1, 1, 1, 1, 10, 1])

    def step(self):
        a = mu(self.s)
        next_s, r, done, _ = self.env.step(a)
        rho = compute_rho(self.s, a)
        delta = r + DISCOUNT * (next_s * self.w).sum() - (self.s * self.w).sum()

        self.w += self.lr * rho * delta * self.s

        self.s = next_s

    def get_weights(self):
        return self.w


class Runner:
    def __init__(self,
                 algo_fn,
                 run,
                 timeout):
        self.algo_fn = algo_fn
        self.run = run
        self.timeout = timeout

    def single_run(self):
        algo = self.algo_fn()
        algo.reset()
        info = []
        for i in range(self.timeout):
            algo.step()
            info.append(algo.get_info())
        return info

    def train(self):
        info = []
        for i in range(self.run):
            info.append(self.single_run())
        return info


def plot(w):
    # w = [run, step, w]
    for p in [0, 6, 7]:
        plot_mean_std(w[:, :, p], label='w%d' % p)
    plt.legend()
    plt.show()


def ProximalETD_expts():
    algo_fn = lambda: ProximalETD(Baird(),
                                  lr1=1,
                                  lr2=0.001,
                                  beta=DISCOUNT,
                                  use_oracle_v=True
                                  )
    info = Runner(algo_fn,
                  10,
                  1000).train()

    thetas = []
    ws = []
    for run_info in info:
        thetas.append([])
        ws.append([])
        for step_info in run_info:
            theta, m_stats, w = step_info
            thetas[-1].append(theta)
            ws[-1].append(w)
        # print(m_stats[6][-100:])
        # plt.plot(m_stats[6])
        # plt.show()
        # print(m_stats[0][-100:])
        # for i in range(7):
        #     print(i, np.std(m_stats[i][-100:]))
    thetas = np.asarray(thetas)
    ws = np.asarray(ws)
    plot(thetas)
    # plot(ws)


def ETD_expts():
    algo_fn = lambda: ETD(Baird(),
                          0.001,
                          use_oracle_m=True,
                          )
    info = Runner(algo_fn,
                  10,
                  10000).train()

    weights = []
    for run_info in info:
        weights.append([])
        for step_info in run_info:
            w, m_stats = step_info
            weights[-1].append(w)
        for i in range(7):
            print(i, np.std(m_stats[i][-100:]))
    weights = np.asarray(weights)
    plot(weights)


def TD_expts():
    algo_fn = lambda: TD(Baird(), 0.01)
    w = Runner(algo_fn,
               30,
               1000).train()
    return w


if __name__ == '__main__':
    ProximalETD_expts()
    # ETD_expts()
    # w = TD_expts()
    # plot(w)
