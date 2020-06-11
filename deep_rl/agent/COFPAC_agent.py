#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class COFPACAgent(BaseAgent):
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
        self.replay = config.replay_fn()

        self.M = torch.zeros((config.num_workers, 1), device=Config.DEVICE)
        self.rho_prev = torch.zeros(self.M.size(), device=Config.DEVICE)

    def eval_episode(self):
        upper_bound = 1000
        t = np.random.randint(upper_bound)
        env = self.config.eval_env
        state = env.reset()
        i = 0
        while i < t:
            i += 1
            action = self.random_action()
            state, _, done, info = env.step(action)

        rewards = []
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            rewards.append(reward[0])
            if done[0]:
                break
        ret = 0
        for r in reversed(rewards):
            ret = r + self.config.discount * ret
        return ret

    def random_action(self):
        config = self.config
        action = [config.eval_env.action_space.sample() for _ in range(config.num_workers)]
        return np.asarray(action)

    def learn_v_m_batch(self, transitions):
        for i in range(self.config.vm_epochs):
            self.learn_v_m(self.replay.sample())
        self.learn_v_m(transitions)

    def learn_v_m(self, transitions):
        config = self.config
        s, a, mu_a, r, next_s, mask = transitions

        prediction = self.network(s, a)
        next_prediction = self.network(next_s)
        prediction_target = self.target_network(s, a)
        next_prediction_target = self.target_network(next_s)

        rho = prediction['pi_a'] / mu_a
        rho = rho.detach()

        v_target = next_prediction_target['v']
        v_target = r + config.discount * mask * v_target
        v_target = v_target.detach()
        td_error = v_target - prediction['v']
        v_loss = td_error.pow(2).mul(0.5).mul(rho.clamp(0, 1)).mean()

        m_target = 1 + config.discount * rho * prediction_target['m']
        m_target = m_target.detach()
        m_next = next_prediction['m'] * mask
        m_loss = (m_target - m_next).pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        (v_loss + m_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def learn(self, s, a, mu_a, r, next_s, mask):
        config = self.config
        self.learn_v_m_batch([s, a, mu_a, r, next_s, mask])

        prediction = self.network(s, a)
        with torch.no_grad():
            target = self.target_network(next_s)['v']
            target = r + config.discount * mask * target
            self.M = mask * config.discount * self.rho_prev * self.M + 1
            rho = prediction['pi_a'] / mu_a
            self.rho_prev = rho
            rho = rho.clamp(0, 2)
            td_error = target - prediction['v']

        entropy = prediction['ent'].mean()

        if config.algo == 'off-pac':
            M = 1
        elif config.algo == 'ace':
            M = self.M
            self.logger.add_histogram('m_ace', self.M, log_level=5)
        elif config.algo == 'cof-pac':
            M = prediction['m'].detach()
            self.logger.add_histogram('m_cof-pac', prediction['m'], log_level=5)
        else:
            raise NotImplementedError
        pi_loss = -M * rho * td_error * prediction['log_pi_a'] - config.entropy_weight * entropy
        pi_loss = pi_loss.mean()

        self.optimizer.zero_grad()
        pi_loss.backward()
        self.optimizer.step()

    def eval_step(self, state):
        with torch.no_grad():
            action = self.network(state)['a']
        if self.config.action_type == 'discrete':
            return to_np(action)
        elif self.config.action_type == 'continuous':
            return to_np(action)
        else:
            raise NotImplementedError

    def step(self):
        config = self.config
        actions = self.random_action()
        if config.action_type == 'discrete':
            mu_a = np.zeros_like(actions) + 1 / config.action_dim
        elif config.action_type == 'continuous':
            mu_a = np.zeros((config.num_workers, )) + 1 / np.power(2.0, config.action_dim)
        else:
            raise NotImplementedError
        next_states, rewards, terminals, info = self.task.step(actions)
        self.record_online_return(info)
        rewards = config.reward_normalizer(rewards)

        mask = (1 - terminals).astype(np.uint8)
        transition = [tensor(self.states),
                      tensor(actions),
                      tensor(mu_a).unsqueeze(1),
                      tensor(rewards).unsqueeze(1),
                      tensor(next_states),
                      tensor(mask).unsqueeze(1)]
        if self.replay.size() >= config.replay_warm_up:
            self.learn(*transition)
        self.replay.feed(transition)
        self.states = next_states

        self.total_steps += config.num_workers

        if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())