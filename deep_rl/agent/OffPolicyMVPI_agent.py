#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class OffPolicyMVPI(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.q_opt = config.q_opt_fn(self.network.parameters())
        self.pi_opt = config.pi_opt_fn(self.network.parameters())
        self.tau_opt = config.tau_opt_fn(self.network.parameters())
        self.replay = config.replay_fn()
        self.total_steps = 0
        self.num_states = 4
        self.avg_reward = 0
        self.pi_a0 = None
        self.compute_pi_a0()
        self.compute_oracle_ratio()
        # self.simulate_oracle()

        if config.repr == 'tabular':
            self.phi = np.eye(self.num_states)
        else:
            raise NotImplementedError
        self.gather_data()

    def compute_pi_a0(self):
        s_0 = np.zeros((1, self.num_states))
        s_0[0, 0] = 1
        pi = self.network(tensor(s_0))['pi']
        pi = to_np(pi).flatten()
        self.pi_a0 = pi[0]

    def compute_oracle_ratio(self):
        config = self.config
        mu_0 = np.zeros((self.num_states + 1, 1))
        mu_0[0, 0] = self.pi_a0
        mu_0[1, 0] = 1 - self.pi_a0
        P_pi = np.zeros((self.num_states + 1, self.num_states + 1))
        P_pi[0, 2] = 0.5
        P_pi[0, 3] = 0.5
        P_pi[1, 4] = 1
        P_pi[2, 0] = P_pi[3, 0] = P_pi[4, 0] = self.pi_a0
        P_pi[2, 1] = P_pi[3, 1] = P_pi[4, 1] = 1 - self.pi_a0

        D = np.eye(self.num_states + 1)
        D = D / D.sum()

        D_inv = np.linalg.inv(D)
        self.tau_star = (1 - config.discount) * np.linalg.inv(
            np.eye(D.shape[0]) - config.discount * D_inv @ P_pi.T @ D) \
                        @ D_inv @ mu_0
        self.d_pi = tensor(D @ self.tau_star)

        r = np.zeros((self.num_states + 1, 1))
        r[2, 0] = 1
        r[3, 0] = -1
        r[4, 0] = -1
        r = r - config.lam * r ** 2 + 2 * config.lam * r * self.avg_reward
        q = np.linalg.inv(
            np.eye(D.shape[0]) - config.discount * P_pi) @ r
        self.q = tensor(q)

    def simulate_oracle(self):
        config = self.config
        episodes = 1000
        len = 100
        gamma = np.power(config.discount, np.arange(len)).reshape(-1, 1)
        stats = np.zeros((episodes, len, self.num_states))
        for ep in range(episodes):
            state = 0
            for t in range(len):
                if state == 0:
                    action = (0 if np.random.rand() < self.pi_a0 else 1)
                else:
                    action = 0
                _, next_state = self.env_step(state, action)
                stats[ep, t, state] = 1
                state = next_state
            prob = stats[: ep + 1].mean(axis=0)
            d_pi = tensor(prob * gamma).sum(0) * (1 - config.discount)
            pi_s0 = tensor([d_pi[0] * self.pi_a0, d_pi[0] * (1 - self.pi_a0)])
            d_pi = torch.cat([pi_s0, d_pi[1:]])
            diff = (d_pi - self.d_pi.view(-1)).pow(2).mean()
            print(diff)

    def env_step(self, state, action):
        if state == 0:
            reward = 0
            if action == 0:
                next_state = np.random.choice([1, 2])
            elif action == 1:
                next_state = 3
            else:
                raise NotImplementedError
        else:
            next_state = 0
            if state == 1:
                reward = 1
            elif state == 2:
                reward = -1
            elif state == 3:
                reward = -1
            else:
                raise NotImplementedError

        return reward, next_state

    def to_phi(self, state):
        return self.phi[state]

    def action_target_policy(self, states):
        actions = np.where(np.random.rand(len(states)) < self.pi_a0, 0, 1)
        actions = np.where(states == 0, actions, 0)
        return actions

    def gather_data(self):
        while not self.replay.full():
            state = np.random.randint(low=0, high=self.num_states)
            if state == 0:
                action = np.random.randint(low=0, high=2)
            else:
                action = 0
            reward, next_state = self.env_step(state, action)
            self.replay.feed([state, action, reward, next_state])

    def sample_mu_0(self, size):
        states_0 = np.zeros(size).astype(np.int)
        actions_0 = self.action_target_policy(states_0)
        states_0 = self.phi[states_0, :]
        return states_0, actions_0

    def reshape(self, x):
        s0 = x[0, :].view(-1)
        s = x[1:, 0].view(-1)
        s = torch.cat([s0, s])
        return s.view(-1, 1)

    def evaluation(self):
        with torch.no_grad():
            phi = tensor(self.phi)
            q = self.network(phi)['q']
            q = self.reshape(q)
            tau = self.network.tau(phi)
            tau = self.reshape(tau)
            tau_loss = (tau - self.tau_star).pow(2).mean()
            q_loss = (q - self.q).pow(2).mean()
        print('q_loss %f, tau_loss %f, pi_a0 %f' % (q_loss, tau_loss, self.pi_a0))
        self.logger.add_scalar('pi_a0', self.pi_a0)

    def density_ratio_learning(self):
        config = self.config

        states, actions, rewards, next_states = self.replay.sample(10)
        next_actions = self.action_target_policy(next_states)
        states = tensor(self.phi[states, :])
        actions = tensor(actions).unsqueeze(-1).long()
        next_states = tensor(self.phi[next_states, :])
        next_actions = tensor(next_actions).unsqueeze(-1).long()

        states_0, actions_0 = self.sample_mu_0(states.shape[0])
        states_0 = tensor(states_0)
        actions_0 = tensor(actions_0).long().unsqueeze(-1)

        tau = self.network.tau(states, actions)
        f = self.network.f(states, actions)
        f_next = self.network.f(next_states, next_actions)
        f_0 = self.network.f(states_0, actions_0)
        u = self.network.u(states.size(0))

        if config.algo == 'GenDICE':
            J_concave = (1 - config.discount) * f_0 + config.discount * tau.detach() * f_next - \
                        tau.detach() * (f + 0.25 * f.pow(2)) + config.dice_lam * (u * tau.detach() - u - 0.5 * u.pow(2))
            J_convex = (1 - config.discount) * f_0.detach() + config.discount * tau * f_next.detach() - \
                       tau * (f.detach() + 0.25 * f.detach().pow(2)) + \
                       config.dice_lam * (u.detach() * tau - u.detach() - 0.5 * u.detach().pow(2))
        elif config.algo == 'GradientDICE':
            J_concave = (1 - config.discount) * f_0 + config.discount * tau.detach() * f_next - \
                        tau.detach() * f - 0.5 * f.pow(2) + config.dice_lam * (u * tau.detach() - u - 0.5 * u.pow(2))
            J_convex = (1 - config.discount) * f_0.detach() + config.discount * tau * f_next.detach() - \
                       tau * f.detach() - 0.5 * f.detach().pow(2) + \
                       config.dice_lam * (u.detach() * tau - u.detach() - 0.5 * u.detach().pow(2))
        elif config.algo == 'DualDICE':
            J_concave = (f.detach() - config.discount * f_next.detach()) * tau - tau.pow(3).mul(1.0 / 3) \
                        - (1 - config.discount) * f_0.detach()
            J_convex = (f - config.discount * f_next) * tau.detach() - tau.detach().pow(3).mul(1.0 / 3) - \
                       (1 - config.discount) * f_0
        else:
            raise NotImplementedError

        J_convex = J_convex.mean()
        loss = J_convex - J_concave.mean()

        self.tau_opt.zero_grad()
        loss.backward()
        self.tau_opt.step()

    def expected_sarsa(self):
        config = self.config
        states, actions, rewards, next_states = self.replay.sample()
        pi_next = []
        for s in next_states:
            if s == 0:
                pi_next.append([self.pi_a0, 1 - self.pi_a0])
            else:
                pi_next.append([1, 0])
        pi_next = tensor(pi_next)
        # next_actions = self.action_target_policy(next_states)
        states = tensor(self.phi[states, :])
        actions = tensor(actions).unsqueeze(-1).long()
        rewards = tensor(rewards).unsqueeze(-1)
        rewards = rewards - config.lam * rewards.pow(2) + 2 * config.lam * rewards * self.avg_reward
        next_states = tensor(self.phi[next_states, :])
        # next_actions = tensor(next_actions).unsqueeze(-1).long()
        # target = self.network(next_states)['q'].gather(1, next_actions)
        target = (self.network(next_states)['q'] * pi_next).sum(-1).unsqueeze(-1)
        target = rewards + config.discount * target
        target = target.detach()
        q = self.network(states)['q'].gather(1, actions)
        loss = (q - target).pow(2).mul(0.5).mean()

        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

    def policy_gradient(self):
        config = self.config
        oracle_tau = []
        oracle_q = []
        exp = []
        # for s, a, _, _ in zip(*self.replay.sample(50)):
        for s, a, _, _ in self.replay.data:
            if s == 0:
                exp.append([s, a])
                oracle_tau.append(self.tau_star[s + a, 0])
                oracle_q.append(self.q[s + a, 0])
        if len(exp) == 0:
            return
        i = np.random.randint(0, len(exp))
        # exp = [exp[i]]
        # oracle_tau = [oracle_tau[i]]
        # oracle_q = [oracle_q[i]]
        oracle_tau = tensor(oracle_tau).unsqueeze(-1)
        oracle_q = tensor(oracle_q).unsqueeze(-1)
        states, actions = zip(*exp)
        states = tensor(self.phi[states, :])
        actions = tensor(actions).unsqueeze(-1).long()

        prediction = self.network(states)
        log_pi = prediction['log_pi'].gather(1, actions)

        if config.use_oracle_q:
            q = oracle_q
        else:
            q = prediction['q'].gather(1, actions).detach()

        if config.use_oracle_ratio:
            tau = oracle_tau
        else:
            tau = self.network.tau(states, actions).detach()
        pi_loss = -(tau * log_pi * q).mean()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

    def compute_avg_reward(self):
        states, actions, rewards, _ = zip(*self.replay.data)
        rewards = tensor(rewards).unsqueeze(-1)
        if self.config.use_oracle_ratio:
            tau = []
            for s, a in zip(states, actions):
                if s == 0:
                    sa = s + a
                else:
                    sa = s + 1
                tau.append(self.tau_star[sa, 0])
            tau = tensor(tau).view(-1, 1)
        else:
            states = tensor(self.phi[states, :])
            actions = tensor(actions).long().unsqueeze(-1)
            tau = self.network.tau(states, actions)

        avg_reward = (tau * rewards).mean()
        self.avg_reward = to_np(avg_reward)

    def step(self):
        self.compute_pi_a0()
        self.compute_oracle_ratio()
        if not self.config.use_oracle_ratio:
            for i in range(100):
                self.density_ratio_learning()
        self.compute_avg_reward()
        self.compute_oracle_ratio()
        for _ in range(100):
            self.expected_sarsa()
        self.policy_gradient()
        self.total_steps += 1

    def eval_episodes(self):
        self.evaluation()