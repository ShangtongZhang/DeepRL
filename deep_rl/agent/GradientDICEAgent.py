#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class GradientDICE(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0

        self.num_states = 13
        self.pi_0 = 0.1
        if config.repr == 'tabular':
            self.phi = np.eye(self.num_states)
        elif config.repr == 'linear':
            self.phi = np.asarray([
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
        else:
            raise NotImplementedError

        if config.discount < 1:
            self.compute_oracle_episodic()
        else:
            self.compute_oracle_continuing()
        print(self.tau_star)
        # self.gather_data()

    def compute_oracle_episodic(self):
        config = self.config
        ind = lambda state, action: state * 2 + action
        P_pi = np.zeros((self.num_states * 2, self.num_states * 2))

        for i in range(0, 2):
            P_pi[ind(i, 0), ind(0, 0)] = self.pi_0
            P_pi[ind(i, 0), ind(0, 1)] = 1 - self.pi_0
            P_pi[ind(i, 1), ind(0, 0)] = self.pi_0
            P_pi[ind(i, 1), ind(0, 1)] = 1 - self.pi_0

        for i in range(2, self.num_states):
            P_pi[ind(i, 0), ind(i - 1, 0)] = self.pi_0
            P_pi[ind(i, 0), ind(i - 1, 1)] = 1 - self.pi_0
            P_pi[ind(i, 1), ind(i - 2, 0)] = self.pi_0
            P_pi[ind(i, 1), ind(i - 2, 1)] = 1 - self.pi_0

        D = np.eye(self.num_states * 2)
        D = D / D.sum()
        mu_0 = np.zeros((self.num_states * 2, 1))
        for i in range(self.num_states):
            mu_0[ind(i, 0)] = self.pi_0
            mu_0[ind(i, 1)] = 1 - self.pi_0
        mu_0 = mu_0 / self.num_states

        D_inv = np.linalg.inv(D)
        self.tau_star = (1 - config.discount) * np.linalg.inv(
            np.eye(D.shape[0]) - config.discount * D_inv @ P_pi.T @ D) \
                        @ D_inv @ mu_0
        self.d_pi = tensor(D @ self.tau_star)
        r1 = np.linalg.matrix_rank(self.phi)
        tau_star = self.tau_star.reshape(-1, 2)
        r2 = np.linalg.matrix_rank(np.concatenate([self.phi, tau_star[:, [0]]], axis=1))
        r3 = np.linalg.matrix_rank(np.concatenate([self.phi, tau_star[:, [1]]], axis=1))
        print('Exactly solvable: %s' % (r1 == r2 and r1 == r3))
        self.tau_star = tensor(self.tau_star)

        X_1 = np.concatenate([self.phi, np.zeros(self.phi.shape)], axis=1).reshape(-1, self.phi.shape[1])
        X_2 = np.concatenate([np.zeros(self.phi.shape), self.phi], axis=1).reshape(-1, self.phi.shape[1])
        X = np.concatenate([X_1, X_2], axis=1)
        A = X.T @ (np.eye(X.shape[0]) - config.discount * P_pi.T) @ D @ X
        print('A is full rank: %s' % (np.linalg.matrix_rank(A) == A.shape[0]))

        self.X = X
        self.D = D
        self.mu_0 = mu_0
        self.P_pi = P_pi

        # self.simulate_tau_star()

    def compute_oracle_continuing(self):
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
        d = P_star[[0], :]
        tau_star = d * self.num_states * 2
        self.tau_star = tensor(tau_star)

        D = np.eye(self.num_states * 2)
        D = D / D.sum()
        X_1 = np.concatenate([self.phi, np.zeros(self.phi.shape)], axis=1).reshape(-1, self.phi.shape[1])
        X_2 = np.concatenate([np.zeros(self.phi.shape), self.phi], axis=1).reshape(-1, self.phi.shape[1])
        X = np.concatenate([X_1, X_2], axis=1)
        A = X.T @ (np.eye(X.shape[0]) - config.discount * P_pi.T) @ D @ X
        print('A is full rank: %s' % (np.linalg.matrix_rank(A) == A.shape[0]))

    def simulate_oracle(self):
        import gym
        config = self.config
        episodes = 1000
        len = 15
        gamma = np.power(config.discount, np.arange(len)).reshape(-1, 1)
        stats = np.zeros((episodes, len, self.num_states))
        env = gym.make('BoyansChainTabular-v0')
        for ep in range(episodes):
            state = np.random.randint(self.num_states)
            env.reset_to(state)
            for t in range(len):
                action = (0 if np.random.rand() < self.pi_0 else 1)
                _, _, _, info = env.step(action)
                stats[ep, t, state] = 1
                state = info['next_s']
            prob = stats[: ep + 1].mean(axis=0)
            d_pi = tensor(prob * gamma).sum(0) * (1 - config.discount)
            d_pi[0] = 1 - d_pi[1:].sum()
            d_pi = d_pi.view(-1, 1)
            d_pi = torch.cat([d_pi * self.pi_0, d_pi * (1 - self.pi_0)], dim=1)
            d_pi = d_pi.contiguous().view(-1)
            diff = (d_pi - self.d_pi.view(-1)).pow(2).mean()
            print(diff)

    def next_state_action(self, state, action):
        if state == 0:
            if self.config.discount < 1:
                next_state = 0
            else:
                next_state = np.random.randint(low=0, high=self.num_states)
        elif state == 1:
            next_state = 0
        else:
            next_state = state - action - 1
        next_action = (0 if np.random.rand() < self.pi_0 else 1)
        return next_state, next_action

    def to_phi(self, state):
        return self.phi[state]

    # def gather_data(self):
    #     while not self.replay.full():
    #         state = np.random.randint(low=0, high=self.num_states)
    #         action = np.random.randint(low=0, high=2)
    #         next_state, next_action = self.next_state_action(state, action)
    #         self.replay.feed([self.to_phi(state), action, self.to_phi(next_state), next_action])

    def sample(self):
        state = np.random.randint(low=0, high=self.num_states)
        action = np.random.randint(low=0, high=2)
        next_state, next_action = self.next_state_action(state, action)
        exp = [self.to_phi(state)], [action], [self.to_phi(next_state)], [next_action]
        exp = [np.asarray(unit) for unit in exp]
        return exp

    def sample_mu_0(self, size):
        states_0 = np.random.randint(low=0, high=self.num_states, size=size)
        actions_0 = np.where(np.random.rand(size) < self.pi_0, 0, 1)
        states_0 = self.phi[states_0, :]
        return states_0, actions_0

    def evaluation(self):
        with torch.no_grad():
            tau = self.network.tau(tensor(self.phi))
            tau = tau.view(-1, 1)
            loss = (tau - self.tau_star).pow(2).mean()
        print(loss)
        self.logger.add_scalar('tau_loss', loss)

    def compute_oracle_f_eta(self):
        config = self.config
        with torch.no_grad():
            tau = self.network.tau(tensor(self.phi))
            tau = to_np(tau.view(-1, 1))
        delta = (1 - config.discount) * self.mu_0 + config.discount * self.P_pi.T @ self.D @ tau - self.D @ tau
        f_star = self.X @ np.linalg.inv(self.X.T @ self.D @ self.X) @ self.X.T @ delta
        eta_star = np.diag(self.D).reshape((-1, 1)).T @ tau - 1
        self.delta = delta
        return f_star, eta_star

    def step(self):
        config = self.config
        states, actions, next_states, next_actions = self.sample()
        states_0, actions_0 = self.sample_mu_0(states.shape[0])

        states = tensor(states)
        actions = tensor(actions).long().unsqueeze(-1)
        next_states = tensor(next_states)
        next_actions = tensor(next_actions).long().unsqueeze(-1)
        states_0 = tensor(states_0)
        actions_0 = tensor(actions_0).long().unsqueeze(-1)

        tau = self.network.tau(states, actions)
        f = self.network.f(states, actions)
        f_next = self.network.f(next_states, next_actions)
        f_0 = self.network.f(states_0, actions_0)
        u = self.network.u(states.size(0))

        if config.algo == 'GenDICE':
            J_concave = (1 - config.discount) * f_0 + config.discount * tau.detach() * f_next - \
                        tau.detach() * (f + 0.25 * f.pow(2)) + config.lam * (u * tau.detach() - u - 0.5 * u.pow(2))
            J_convex = (1 - config.discount) * f_0.detach() + config.discount * tau * f_next.detach() - \
                       tau * (f.detach() + 0.25 * f.detach().pow(2)) + \
                       config.lam * (u.detach() * tau - u.detach() - 0.5 * u.detach().pow(2))
        elif config.algo == 'GradientDICE':
            # f_star, u_star = self.compute_oracle_f_eta()
            # u_oracle = tensor(u_star)
            # f_star = tensor(f_star).view(-1, 2)
            # f_oracle = f_star[np.asscalar(to_np(states.argmax(1))), np.asscalar(to_np(actions))]
            # f_next_oracle = f_star[np.asscalar(to_np(next_states.argmax(1))), np.asscalar(to_np(next_actions))]
            # f_0_oracle = f_star[np.asscalar(to_np(states_0.argmax(1))), np.asscalar(to_np(actions_0))]
            #
            # with torch.no_grad():
            #     learned_f = self.network.f(tensor(self.phi))
            #     dual_loss = (learned_f - f_star).pow(2).mean()
            #     self.logger.add_scalar('dual_loss', dual_loss, log_level=5)
            #     learned_f = to_np(learned_f.view(-1, 1))
            #     J_concave_analytical = self.delta.T @ learned_f - 0.5 * learned_f.T @ self.D @ learned_f
            #     self.logger.add_scalar('J_concave_analytical', J_concave_analytical, log_level=5)
            #     max_J_concave = 0.5 * self.delta.T @ self.X @ np.linalg.inv(self.X.T @ self.D @ self.X) @ self.X.T @ self.delta
            #     # f_star = to_np(f_star.view(-1, 1))
            #     # max_J_concave_2 = self.delta.T @ f_star - 0.5 * f_star.T @ self.D @ f_star
            #     self.logger.add_scalar('max_J_concave', max_J_concave, log_level=5)
            #     # self.logger.add_scalar('max_J_concave_2', max_J_concave_2, log_level=5)
            #
            # if config.oracle_dual:
            #     u, f, f_next, f_0 = u_oracle, f_oracle, f_next_oracle, f_0_oracle

            J_concave = (1 - config.discount) * f_0 + config.discount * tau.detach() * f_next - \
                        tau.detach() * f - 0.5 * f.pow(2) + config.lam * (u * tau.detach() - u - 0.5 * u.pow(2))
            # f = self.network.f(tensor(self.phi)).view(-1, 1)
            # J_concave = tensor(self.delta).t() @ f - 0.5 * f.t() @ tensor(self.D) @ f
            J_convex = (1 - config.discount) * f_0.detach() + config.discount * tau * f_next.detach() - \
                       tau * f.detach() - 0.5 * f.detach().pow(2) + \
                       config.lam * (u.detach() * tau - u.detach() - 0.5 * u.detach().pow(2))

            self.logger.add_scalar('J_concave_sample', J_concave.mean(), log_level=5)

        elif config.algo == 'DualDICE':
            J_concave = (f.detach() - config.discount * f_next.detach()) * tau - tau.pow(3).mul(1.0 / 3) \
                        - (1 - config.discount) * f_0.detach()
            J_convex = (f - config.discount * f_next) * tau.detach() - tau.detach().pow(3).mul(1.0 / 3) - \
                       (1 - config.discount) * f_0
        else:
            raise NotImplementedError

        J_convex = J_convex.mean() + config.ridge * self.network.ridge()
        loss = J_convex - J_concave.mean()

        self.total_steps += 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval_episodes(self):
        self.evaluation()
