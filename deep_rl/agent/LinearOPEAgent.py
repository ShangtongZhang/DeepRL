#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class LinearOPEAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.discount = 1
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.rates = deque(maxlen=100)
        self.GQ2_cache = None

        self.num_states = 13
        self.pi_0 = config.pi0
        self.mu_0 = config.mu0
        if config.repr == 'linear':
            self.phi_s = np.asarray([
                [0, 0, 0, 1],
                [0, 0, 0.25, 0.75],
                [0, 0, 0.5, 0.5],
                [0, 0, 0.75, 0.25],
                [0, 0, 1, 0],
                [0, 0.25, 0.75, 0],
                [0, 0.5, 0.5, 0],
                [0, 0.75, 0.25, 0],
                [0, 1, 0, 0],
                [0.25, 0.75, 0, 0],
                [0.5, 0.5, 0, 0],
                [0.75, 0.25, 0, 0],
                [1, 0, 0, 0],
            ])
            self.phi_a = np.eye(2)
            self.rewards = [1, 2]
        else:
            raise NotImplementedError

        self.compute_oracle()
        # self.simulate_oracle()


    def compute_oracle(self):
        config = self.config
        ind = lambda state, action: state * 2 + action
        P_pi = np.zeros((self.num_states * 2, self.num_states * 2))

        for j in range(0, self.num_states):
            P_pi[ind(0, 0), ind(j, 0)] = self.pi_0 / self.num_states
            P_pi[ind(0, 0), ind(j, 1)] = (1 - self.pi_0) / self.num_states
            P_pi[ind(0, 1), ind(j, 0)] = self.pi_0 / self.num_states
            P_pi[ind(0, 1), ind(j, 1)] = (1 - self.pi_0) / self.num_states

        for i in range(1, 2):
            P_pi[ind(i, 0), ind(0, 0)] = self.pi_0
            P_pi[ind(i, 0), ind(0, 1)] = 1 - self.pi_0
            P_pi[ind(i, 1), ind(0, 0)] = self.pi_0
            P_pi[ind(i, 1), ind(0, 1)] = 1 - self.pi_0

        for i in range(2, self.num_states):
            P_pi[ind(i, 0), ind(i - 1, 0)] = self.pi_0
            P_pi[ind(i, 0), ind(i - 1, 1)] = 1 - self.pi_0
            P_pi[ind(i, 1), ind(i - 2, 0)] = self.pi_0
            P_pi[ind(i, 1), ind(i - 2, 1)] = 1 - self.pi_0

        N = 10000
        P = np.eye(self.num_states * 2)
        P_star = 0
        for _ in range(N):
            P_star += P
            P = P @ P_pi
        P_star = P_star / float(N)
        r = np.tile(self.rewards, self.num_states)
        self.rate_oracle = np.sum(P_star[0, :] * r)

        D = np.eye(self.num_states * 2)
        X = np.zeros((self.num_states * 2, self.phi_s.shape[1] + self.phi_a.shape[1]))
        for i in range(self.num_states):
            X[2 * i] = np.concatenate([self.phi_s[i], self.phi_a[0]])
            X[2 * i + 1] = np.concatenate([self.phi_s[i], self.phi_a[1]])
            D[2 * i, 2 * i] = self.mu_0 / self.num_states
            D[2 * i + 1, 2 * i + 1] = (1 - self.mu_0) / self.num_states
        A = X.T @ D @ (np.eye(X.shape[0]) - P_pi) @ X
        print('A is full rank: %s' % (np.linalg.matrix_rank(A) == A.shape[0]))

    def simulate_oracle(self):
        rewards = []
        state, action = self.sample_from_d_mu()
        while len(rewards) < 1000000:
            reward, state, action = self.act(state, action)
            rewards.append(reward)
        assert np.abs(self.rate_oracle - np.mean(rewards)) < 1e-3

    def sample_from_d_mu(self):
        state = np.random.randint(low=0, high=self.num_states)
        action = (0 if np.random.rand() < self.mu_0 else 1)
        return state, action

    def act(self, state, action):
        if state == 0:
            next_state = np.random.randint(low=0, high=self.num_states)
        elif state == 1:
            next_state = 0
        else:
            next_state = state - action - 1
        next_action = (0 if np.random.rand() < self.pi_0 else 1)
        reward = self.rewards[action]
        return reward, next_state, next_action

    def to_phi(self, state, action):
        return np.concatenate([self.phi_s[state], self.phi_a[action]])

    def sample(self):
        state, action = self.sample_from_d_mu()
        reward, next_state, next_action = self.act(state, action)
        exp = [self.to_phi(state, action)], [reward], [self.to_phi(next_state, next_action)]
        exp = [np.asarray(unit) for unit in exp]
        return exp

    def evaluation(self):
        rate = np.mean(self.rates)
        if np.isnan(rate):
            return
        rate_loss = np.abs(self.rate_oracle - rate)
        self.logger.add_scalar('rate_loss', rate_loss)

    def step(self):
        config = self.config
        sa, r, next_sa = self.sample()
        sa = tensor(sa)
        r = tensor(r).unsqueeze(-1)
        next_sa = tensor(next_sa)

        ex_sa = torch.cat([torch.ones((sa.size(0), 1)), sa], dim=1)
        ex_next_sa = torch.cat([torch.ones((next_sa.size(0), 1)), next_sa], dim=1)
        e1 = torch.cat([torch.ones((sa.size(0), 1)), torch.zeros(sa.size())], dim=1)
        loss = None
        rate = None

        if config.algo in ['GenDICE', 'GradientDICE']:
            tau = self.network.tau(sa)
            f = self.network.f(sa)
            f_next = self.network.f(next_sa)
            u = self.network.u(sa.size(0))
            rate = self.network.rate()
            if config.algo == 'GenDICE':
                J_concave = config.discount * tau.detach() * f_next - \
                            tau.detach() * (f + 0.25 * f.pow(2)) + config.lam * (u * tau.detach() - u - 0.5 * u.pow(2))
                J_convex = config.discount * tau * f_next.detach() - \
                           tau * (f.detach() + 0.25 * f.detach().pow(2)) + \
                           config.lam * (u.detach() * tau - u.detach() - 0.5 * u.detach().pow(2))
            elif config.algo == 'GradientDICE':
                J_concave = config.discount * tau.detach() * f_next - \
                            tau.detach() * f - 0.5 * f.pow(2) + config.lam * (u * tau.detach() - u - 0.5 * u.pow(2))
                J_convex = config.discount * tau * f_next.detach() - \
                           tau * f.detach() - 0.5 * f.detach().pow(2) + \
                           config.lam * (u.detach() * tau - u.detach() - 0.5 * u.detach().pow(2))
            else:
                raise NotImplementedError
            r_loss = (tau.detach() * r - rate).pow(2).mul(0.5).mean()
            J_convex = J_convex.mean() + config.ridge * self.network.ridge() + r_loss
            loss = J_convex - J_concave.mean()
        elif config.algo == 'FQE':
            target = r - self.network.u(e1) + self.network.u(ex_next_sa)
            target = target.detach()
            loss = (target - self.network.u(ex_sa)).pow(2).mul(0.5)
            rate = self.network.rate()
        elif config.algo == 'GQ1':
            td_error = r - self.network.u(e1) + self.network.u(ex_next_sa) - self.network.u(ex_sa)
            Ynu = self.network.nu(ex_sa)
            J_concave = 2 * Ynu * td_error.detach() - Ynu.pow(2)
            J_convex = 2 * Ynu.detach() * td_error + config.ridge * self.network.ridge()
            loss = J_convex.mean() - J_concave.mean()
            rate = self.network.rate()
        elif config.algo == 'GQ2':
            if self.GQ2_cache is None:
                self.GQ2_cache = [sa, r, next_sa]
            else:
                sa1, r1, next_sa1 = self.GQ2_cache
                self.GQ2_cache = None
                rate1 = r1 + self.network.w(next_sa1) - self.network.w(sa1)
                rate2 = r + self.network.w(next_sa) - self.network.w(sa)
                td_error = rate2 - rate1
                Xnu = self.network.nu(sa)
                J_concave = 2 * Xnu * td_error.detach() - Xnu.pow(2)
                J_convex = 2 * Xnu.detach() * td_error + config.ridge * self.network.ridge()
                rate = self.network.rate()
                r_loss = ((rate1 + rate2).mul(0.5).detach() - rate).pow(2).mul(0.5).mean()
                loss = J_convex.mean() + r_loss - J_concave.mean()
        else:
            raise NotImplementedError

        if self.total_steps:
            if rate is None:
                rate = self.rates[-1]
            else:
                rate = np.asscalar(rate.detach().numpy())
            self.rates.append(rate)

        self.total_steps += 1
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval_episodes(self):
        self.evaluation()
