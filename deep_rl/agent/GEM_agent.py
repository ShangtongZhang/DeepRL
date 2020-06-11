#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class COFPACBairdAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

        self.P_pi_solid = np.zeros((7, 7))
        self.P_pi_solid[:, -1] = 1
        self.P_pi_dashed = np.zeros((7, 7))
        self.P_pi_dashed[:, :-1] = 1.0 / 6

        self.r_solid = np.zeros((7, 1))
        self.r_dashed = np.ones((7, 1))
        # self.r_dashed = np.zeros((7, 1))

        self.M = 0

        self.u = np.random.randn(config.eval_env.state_dim)
        self.w = np.random.randn(config.eval_env.state_dim)

        self.J_pi_loss = deque(maxlen=config.loss_interval)

    def oracle_info(self, w):
        config = self.config
        env = self.config.eval_env.env.envs[0].env
        phi = env.phi
        self.phi = phi
        pi = to_np(self.network(phi))

        P_pi = pi[:, [0]] * self.P_pi_solid + pi[:, [1]] * self.P_pi_dashed

        d_mu = np.ones((7, 1), dtype=np.float32) / 7
        D_mu = np.diag(d_mu.flatten())
        I = np.eye(7)
        m = np.linalg.inv(D_mu) @ np.linalg.inv(I - config.discount * P_pi.T) @ d_mu

        self.P_pi = P_pi
        self.D_mu = D_mu

        r = pi[:, [0]] * self.r_solid + pi[:, [1]] * self.r_dashed
        v = np.linalg.inv(I - config.discount * P_pi) @ r

        m_hat = phi @ w.reshape((phi.shape[1], 1))
        delta_w_bar = np.ones((7, 1)) + config.discount * np.linalg.inv(D_mu) @ P_pi.T @ D_mu @ m_hat - m_hat

        tmp = phi.T @ D_mu @ delta_w_bar
        # oracle_u = (np.linalg.inv(phi.T @ D_mu @ phi)) @ tmp
        # J = 0.5 * tmp.T @ oracle_u

        # grad = phi.T @ D_mu @ (config.discount * np.linalg.inv(D_mu) @ P_pi.T @ D_mu @ phi - phi)
        # grad_w = grad.T @ oracle_u

        A = phi.T @ (np.eye(7) - config.discount * P_pi.T) @ D_mu @ phi
        # C = phi.T @ D_mu @ phi
        b = phi.T @ d_mu
        # G = np.block([[-C, -A], [A.T, np.zeros((7, 7))]])
        # g = np.block([[b], [np.zeros((7, 1))]])
        # w_oracle = -np.linalg.inv(G) @ g
        # w_oracle = np.linalg.inv(A) @ b
        # w_oracle = w_oracle.flatten()[7:]

        J_pi = (d_mu * v).sum()

        q_solid = self.r_solid + config.discount * (self.P_pi_solid * v.T).sum(1, keepdims=True)
        q_dashed = self.r_dashed + config.discount * (self.P_pi_dashed * v.T).sum(1, keepdims=True)
        q = np.concatenate([q_solid, q_dashed], axis=1)

        return dict(
            m=m.flatten(),
            v=v.flatten(),
            J=0,
            # u=oracle_u.flatten(),
            # J=J,
            # grad_w=grad_w.flatten(),
            # w_oracle=w_oracle.flatten(),
            J_pi=J_pi,
            q=q,
            d_mu=d_mu,
        )

    def step(self):
        config = self.config

        oracle = self.oracle_info(self.w)

        pi = self.network(self.states)
        self.logger.add_scalar('pi_0', pi[0, 0], log_level=5)
        mu = config.mu()

        dist = torch.distributions.Categorical(probs=mu)
        action = dist.sample()
        action = action.detach()

        rho = (pi / mu).gather(1, action.unsqueeze(-1))
        rho = rho.detach()

        next_states, rewards, terminals, info = self.task.step(to_np(action))
        info = info[0]

        oracle_m = oracle['m'][info['s']]
        adv = oracle['q'][info['s'], np.asscalar(to_np(action))] - oracle['v'][info['s']]

        if config.m_type == 'oracle':
            m = oracle_m
        elif config.m_type == 'gem':
            delta_u = 1 + config.discount * np.asscalar(to_np(rho)) * (
                    self.states[0] * self.w).sum() - (next_states[0] * self.w).sum() - (next_states[0] * self.u).sum()
            delta_u = delta_u * next_states[0]
            # self.w -= config.lr_m * oracle['grad_w']
            # delta_w = (next_states[0] * oracle['u']).sum() * (
            #         next_states[0] - config.discount * np.asscalar(to_np(rho)) * self.states[0])
            delta_w = (next_states[0] * self.u).sum() * (
                    next_states[0] - config.discount * np.asscalar(to_np(rho)) * self.states[0])
            self.w += config.lr_m * delta_w
            self.u += config.lr_m * delta_u
            m = (self.states[0] * self.w).sum()
            m_loss = np.abs(m - oracle_m)
            if info['s'] != 6:
                self.logger.add_scalar('m_loss', m_loss, log_level=5)
            if info['s'] == 6:
                self.logger.add_scalar('m_hat_6', m, log_level=5)
            else:
                self.logger.add_scalar('m_hat_other', m, log_level=5)
            # self.logger.add_scalar('J', oracle['J'], log_level=5)
        elif config.m_type == 'trace':
            self.M = 1 + config.discount * np.asscalar(to_np(rho)) * self.M
            m = self.M
            self.logger.add_scalar('trace_loss', np.abs(m - oracle_m), log_level=5)
        else:
            raise NotImplementedError

        self.J_pi_loss.append(np.asscalar(oracle['J_pi']))
        if self.total_steps % config.loss_interval == 0:
            self.logger.add_scalar('J_pi', np.mean(self.J_pi_loss))

        if self.total_steps % config.pi_delay == 0:
            # if config.oracle_pi_grad:
            #     pi = self.network(self.phi)
            #     m = tensor(oracle['m']).view(-1, 1)
            #     q = tensor(oracle['q'])
            #     d_mu = tensor(oracle['d_mu']).view(-1, 1)
            #     # pi_loss = -(d_mu * m * q).detach() * pi
            #     # pi_loss = pi_loss.sum()
            #
            #     # pi_loss = -(m * q).detach() * pi
            #     # pi_loss = pi_loss[info['s'], :].sum()
            #
            #     pi_loss = -(m * q).detach() * pi.add(1e-5).log()
            #     pi_loss = pi_loss[info['s'], np.asscalar(to_np(action))] * rho.detach()
            # else:
            pi_loss = - m * adv * rho * pi.gather(1, action.unsqueeze(-1)).add(1e-5).log()
            self.optimizer.zero_grad()
            pi_loss.backward()
            self.optimizer.step()

        self.states = next_states
        self.total_steps += 1


class GEMAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.total_steps = 0
        self.states = self.task.reset()

        self.P_pi_solid = np.zeros((7, 7))
        self.P_pi_solid[:, -1] = 1
        self.P_pi_dashed = np.zeros((7, 7))
        self.P_pi_dashed[:, :-1] = 1.0 / 6

        self.r_solid = np.zeros((7, 1))
        self.r_dashed = np.ones((7, 1))
        # self.r_dashed = np.zeros((7, 1))

        self.M = 0
        self.rho = 0

        # GEM parameters
        self.u = np.random.randn(config.eval_env.state_dim)
        self.w = np.random.randn(config.eval_env.state_dim)

        # ETD parameters
        self.d = np.random.randn(config.eval_env.state_dim)

        self.trace_loss = deque(maxlen=config.loss_interval)
        self.gem_loss = deque(maxlen=config.loss_interval)
        self.rmsve_loss = deque(maxlen=config.loss_interval)
        self.etd_weight = deque(maxlen=config.loss_interval)

    def oracle_info(self, w):
        config = self.config
        env = self.config.eval_env.env.envs[0].env
        phi = env.phi
        self.phi = phi
        pi = self.pi

        P_pi = pi[:, [0]] * self.P_pi_solid + pi[:, [1]] * self.P_pi_dashed

        d_mu = np.ones((7, 1), dtype=np.float32) / 7
        D_mu = np.diag(d_mu.flatten())
        I = np.eye(7)
        m = np.linalg.inv(D_mu) @ np.linalg.inv(I - config.discount * P_pi.T) @ d_mu

        self.P_pi = P_pi
        self.D_mu = D_mu
        self.d_mu = d_mu

        r = pi[:, [0]] * self.r_solid + pi[:, [1]] * self.r_dashed
        v = np.linalg.inv(I - config.discount * P_pi) @ r

        m_hat = phi @ w.reshape((phi.shape[1], 1))
        delta_w_bar = np.ones((7, 1)) + config.discount * np.linalg.inv(D_mu) @ P_pi.T @ D_mu @ m_hat - m_hat

        tmp = phi.T @ D_mu @ delta_w_bar
        # oracle_u = (np.linalg.inv(phi.T @ D_mu @ phi)) @ tmp
        # J = 0.5 * tmp.T @ oracle_u

        # grad = phi.T @ D_mu @ (config.discount * np.linalg.inv(D_mu) @ P_pi.T @ D_mu @ phi - phi)
        # grad_w = grad.T @ oracle_u

        A = phi.T @ (np.eye(7) - config.discount * P_pi.T) @ D_mu @ phi
        # C = phi.T @ D_mu @ phi
        b = phi.T @ d_mu
        # G = np.block([[-C, -A], [A.T, np.zeros((7, 7))]])
        # g = np.block([[b], [np.zeros((7, 1))]])
        # w_oracle = -np.linalg.inv(G) @ g
        # w_oracle = np.linalg.inv(A) @ b
        # w_oracle = w_oracle.flatten()[7:]

        J_pi = (d_mu * v).sum()

        q_solid = self.r_solid + config.discount * (self.P_pi_solid * v.T).sum(1, keepdims=True)
        q_dashed = self.r_dashed + config.discount * (self.P_pi_dashed * v.T).sum(1, keepdims=True)
        q = np.concatenate([q_solid, q_dashed], axis=1)

        return dict(
            m=m.flatten(),
            v=v.flatten(),
            J=0,
            # u=oracle_u.flatten(),
            # J=J,
            # grad_w=grad_w.flatten(),
            # w_oracle=w_oracle.flatten(),
            J_pi=J_pi,
            q=q,
            d_mu=d_mu,
        )

    def update_pi(self):
        config = self.config
        self.pi_solid = np.asarray([[config.pi_solid, 1 - config.pi_solid] for _ in range(7)])
        # self.pi_solid = np.random.rand()
        self.pi = np.concatenate([self.pi_solid, 1 - self.pi_solid], axis=1)

    def test_grad_w(self):
        oracle = self.oracle_info(self.w)
        eps = 1e-5
        for i in range(len(self.w)):
            w = np.copy(self.w)
            w[i] += eps
            o = self.oracle_info(w)
            d_i = (o['J'] - oracle['J']) / eps
            print(d_i - oracle['grad_w'][i])

    def torch_oracle(self, w):
        config = self.config
        phi = tensor(self.phi)
        P_pi = tensor(self.P_pi)
        D_mu = tensor(self.D_mu)

        w = tensor(w).view(7, 1).requires_grad_()
        m_hat = phi @ w
        delta_w_bar = torch.ones((7, 1)) + config.discount * torch.inverse(D_mu) @ P_pi.t() @ D_mu @ m_hat - m_hat

        tmp = phi.t() @ D_mu @ delta_w_bar
        oracle_u = torch.inverse(phi.t() @ D_mu @ phi) @ tmp
        J = 0.5 * tmp.t() @ oracle_u

        J = J.sum()
        J.backward()

        return {
            'J': J,
            'grad_w': w.grad,
        }

    def step(self):
        config = self.config
        if self.total_steps % config.pi_duration == 0:
            self.update_pi()

        # self.test_grad_w()

        oracle = self.oracle_info(self.w)
        # o_ = self.oracle_info(oracle['w_oracle'])
        # o_ = self.torch_oracle(self.w)

        mu = config.mu()
        action = np.random.choice([0, 1], p=mu)

        next_states, rewards, terminals, info = self.task.step([action])
        info = info[0]

        s = info['s']
        next_s = info['next_s']

        rho = self.pi[s, action] / mu[action]

        oracle_m = oracle['m'][s]

        delta_u = 1 + config.discount * rho * (
                self.states[0] * self.w).sum() - (next_states[0] * self.w).sum() - (next_states[0] * self.u).sum()
        delta_u = delta_u * next_states[0]
        # self.w -= config.lr_m * oracle['grad_w']
        # self.w -= config.lr_m * to_np(o_['grad_w']).flatten()
        # self.logger.add_scalar('grad_w', np.linalg.norm(oracle['grad_w']))
        # self.logger.add_scalar('error_w', np.linalg.norm(oracle['w_oracle'] - self.w))
        # delta_w = (next_states[0] * oracle['u']).sum() * (
        #         next_states[0] - config.discount * rho * self.states[0])
        delta_w = (next_states[0] * self.u).sum() * (
                next_states[0] - config.discount * rho * self.states[0])
        self.w += config.lr_m * delta_w
        self.u += config.lr_m * delta_u
        m = (self.states[0] * self.w).sum()
        gem_loss = np.abs(m - oracle_m)

        if s == 6:
            self.logger.add_scalar('gem_loss_s6', gem_loss, log_level=5)
        else:
            self.logger.add_scalar('gem_loss_other', gem_loss, log_level=5)

        self.logger.add_scalar('J_w', oracle['J'], log_level=5)
        self.gem_loss.append(gem_loss)
        if self.total_steps % config.loss_interval == 0:
            self.logger.add_scalar('gem_loss', np.mean(self.gem_loss))

        self.M = 1 + config.discount * self.rho * self.M
        self.trace_loss.append(np.abs(self.M - oracle_m))
        if self.total_steps % config.loss_interval == 0:
            self.logger.add_scalar('trace_loss', np.mean(self.trace_loss))

        td_error = rewards[0] + config.discount * (next_states[0] * self.d).sum() - (self.states[0] * self.d).sum()

        if config.m_type == 'oracle':
            m_t = oracle_m
        elif config.m_type == 'trace':
            m_t = self.M
        elif config.m_type == 'gem':
            m_t = m
        else:
            raise NotImplementedError

        if config.etd:
            self.d += config.lr_etd * rho * td_error * m_t * self.states[0]
            ve = (self.phi * self.d.reshape(1, -1)).sum(1) - oracle['v']
            rmsve = np.sqrt((np.power(ve, 2) * self.d_mu).sum())
            self.rmsve_loss.append(rmsve)
            self.etd_weight.append(np.linalg.norm(self.d))
            if self.total_steps % config.loss_interval == 0:
                self.logger.add_scalar('RMSVE', np.mean(self.rmsve_loss))
                # self.logger.add_scalar('etd_weight', np.mean(self.etd_weight))

        self.rho = rho
        self.states = next_states
        self.total_steps += 1
