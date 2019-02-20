#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class GeoffPACAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

        self.episode_rewards = []
        self.online_rewards = np.zeros(config.num_workers)

        self.F1 = torch.zeros((config.num_workers, 1), device=Config.DEVICE)
        self.F2 = Grads(self.network, config.num_workers)
        self.grad_prev = Grads(self.network, config.num_workers)
        self.rho_prev = torch.zeros(self.F1.size(), device=Config.DEVICE)
        self.c_prev = torch.zeros(self.F1.size(), device=Config.DEVICE)

    def random_action(self):
        config = self.config
        return np.random.randint(0, config.action_dim, size=(config.num_workers,))

    def off_pac_update(self, s, a, mu_a, r, next_s, m):
        config = self.config
        prediction = self.network(s, a)
        with torch.no_grad():
            target = self.target_network(next_s)['v']
            target = r + config.discount * m * target
            rho = prediction['pi_a'] / mu_a
        td_error = target - prediction['v']
        config.logger.add_histogram('v', prediction['v'])
        v_loss = td_error.pow(2).mul(0.5).mul(rho.clamp(0, 1)).mean()
        entropy = prediction['ent'].mean()
        pi_loss = -rho * td_error.detach() * prediction['log_pi_a'] - config.entropy_weight * entropy
        pi_loss = pi_loss.mean()

        loss = v_loss + pi_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

    def ace_update(self, s, a, mu_a, r, next_s, m):
        config = self.config
        prediction = self.network(s, a)
        with torch.no_grad():
            target = self.target_network(next_s)['v']
            target = r + config.discount * m * target
            self.F1 = m * config.discount * self.rho_prev * self.F1 + 1
            M = (1 - config.lam1) + config.lam1 * self.F1
            rho = prediction['pi_a'] / mu_a

        td_error = target - prediction['v']
        config.logger.add_histogram('v', prediction['v'])
        v_loss = td_error.pow(2).mul(0.5).mul(rho.clamp(0, 1)).mean()

        entropy = prediction['ent'].mean()
        pi_loss = -M * rho * td_error.detach() * prediction['log_pi_a'] - config.entropy_weight * entropy
        pi_loss = pi_loss.mean()

        self.rho_prev = rho

        loss = (v_loss + pi_loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

    def geoff_pac_update(self, s, a, mu_a, r, next_s, m):
        config = self.config
        prediction = self.network(s, a)
        next_prediction = self.network(next_s)
        prediction_target = self.target_network(s, a)
        next_prediction_target = self.target_network(next_s)

        rho = prediction['pi_a'] / mu_a
        rho = rho.detach()

        v_target = next_prediction_target['v']
        v_target = r + config.discount * m * v_target
        v_target = v_target.detach()
        td_error = v_target - prediction['v']
        config.logger.add_histogram('v', prediction['v'])
        config.logger.add_histogram('c', prediction['c'])
        v_loss = td_error.pow(2).mul(0.5).mul(rho.clamp(0, 1)).mean()

        c_target = config.gamma_hat * rho.clamp(0, 2) * prediction_target['c'] + 1 - config.gamma_hat
        c_target = c_target.detach()
        c_next = next_prediction['c'] * m

        c_normalizer = (c_next.sum(-1).unsqueeze(-1) - c_next).mul(1 / (c_next.size(0) - 1)) - 1
        c_normalizer = c_normalizer.detach() * c_next

        c_loss = (c_target - c_next).pow(2).mul(0.5).mean() + config.c_coef * c_normalizer.mean()

        self.optimizer.zero_grad()
        (v_loss + c_loss).backward()
        # nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        prediction = self.network(s, a)
        c = prediction['c'].detach().clamp(0, 2)
        self.F1 = m * config.discount * self.rho_prev * self.F1 + c
        M1 = (1 - config.lam1) * c + config.lam1 * self.F1

        I = self.grad_prev
        I.mul(self.rho_prev * self.c_prev)
        self.F2.mul(config.gamma_hat * self.rho_prev).add(I)

        v = prediction['v'].detach()
        M2 = I
        F2 = self.F2.clone()
        F2.mul(config.lam2)
        M2.mul(1 - config.lam2).add(F2).mul(config.gamma_hat * v)

        log_pi_a = prediction['log_pi_a'].squeeze(-1)
        self.grad_prev = Grads(self.network, config.num_workers)
        for i in range(config.num_workers):
            self.optimizer.zero_grad()
            log_pi_a[i].backward(retain_graph=True)
            self.grad_prev.grads[i].add(self.network)

        grad = self.grad_prev.clone()
        grad.mul(rho.clamp(0, 2) * M1 * td_error.detach()).add(M2)
        grad = grad.mean().mul(-0.1)

        self.optimizer.zero_grad()
        grad.assign(self.network)
        # nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        self.rho_prev = rho.clamp(0, 2)
        self.c_prev = c

    def eval_step(self, state):
        with torch.no_grad():
            action = self.network(state)['a']
        return np.asscalar(to_np(action))

    def step(self):
        config = self.config
        actions = self.random_action()
        mu_a = np.zeros_like(actions) + 1 / config.action_dim
        next_states, rewards, terminals, _ = self.task.step(actions)
        self.online_rewards += rewards
        rewards = config.reward_normalizer(rewards)
        for i, terminal in enumerate(terminals):
            if terminals[i]:
                self.episode_rewards.append(self.online_rewards[i])
                self.online_rewards[i] = 0

        mask = (1 - terminals).astype(np.uint8)
        transition = [tensor(self.states),
                      tensor(actions),
                      tensor(mu_a).unsqueeze(1),
                      tensor(rewards).unsqueeze(1),
                      tensor(next_states),
                      tensor(mask).unsqueeze(1)]
        if config.algo == 'off-pac':
            self.off_pac_update(*transition)
        elif config.algo == 'ace':
            self.ace_update(*transition)
        elif config.algo == 'geoff-pac':
            self.geoff_pac_update(*transition)
        else:
            raise NotImplementedError

        self.states = next_states

        self.total_steps += 1

        if self.total_steps % config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
