#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class OptionCriticAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []

        self.total_steps = 0
        self.worker_index = tensor(np.arange(config.num_workers)).long()

        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def sample_option(self, prediction, epsilon, prev_option, is_intial_states):
        with torch.no_grad():
            q_option = prediction['q']
            pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
            greedy_option = q_option.argmax(dim=-1, keepdim=True)
            prob = 1 - epsilon + epsilon / q_option.size(1)
            prob = torch.zeros_like(pi_option).add(prob)
            pi_option.scatter_(1, greedy_option, prob)

            mask = torch.zeros_like(q_option)
            mask[:, prev_option] = 1
            beta = prediction['beta']
            pi_hat_option = (1 - beta) * mask + beta * pi_option

            dist = torch.distributions.Categorical(probs=pi_option)
            options = dist.sample()
            dist = torch.distributions.Categorical(probs=pi_hat_option)
            options_hat = dist.sample()

            options = torch.where(is_intial_states, options, options_hat)
        return options

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init'])

        for _ in range(config.rollout_length):
            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction['pi'] = prediction['pi'][self.worker_index, options]
            prediction['log_pi'] = prediction['log_pi'][self.worker_index, options]
            dist = torch.distributions.Categorical(probs=prediction['pi'])
            actions = dist.sample()
            entropy = dist.entropy()

            next_states, rewards, terminals, _ = self.task.step(to_np(actions))
            next_states = config.state_normalizer(next_states)
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         'ent': entropy.unsqueeze(-1),
                         'a': actions.unsqueeze(-1),
                         'init': self.is_initial_states.unsqueeze(-1).float()})

            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            prediction = self.target_network(self.states)
            storage.add(prediction)
            storage.placeholder()
            betas = prediction['beta'][self.worker_index, self.prev_options]
            ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
                  betas * torch.max(prediction['q'], dim=-1)[0]
            ret = ret.unsqueeze(-1)

        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            adv = ret - storage.q[i].gather(1, storage.o[i])
            storage.ret[i] = ret
            storage.adv[i] = adv

            v = storage.q[i].max(dim=-1, keepdim=True)[0]
            q = storage.q[i].gather(1, storage.prev_o[i])
            storage.beta_adv[i] = q - v + config.termination_regularizer

        q, beta, log_pi, ret, adv, beta_adv, ent, option, action, initial_states = \
            storage.cat(['q', 'beta', 'log_pi', 'ret', 'adv', 'beta_adv', 'ent', 'o', 'a', 'init'])

        q_loss = (q.gather(1, option).unsqueeze(-1) - ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(log_pi.gather(1, action).unsqueeze(-1) * adv.detach()) - config.entropy_weight * ent
        pi_loss = pi_loss.mean()
        beta_loss = (betas * beta_adv.detach() * (1 - initial_states)).mean()

        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()


        # for i in reversed(range(len(rollout))):
        #     q_options, betas, options, prev_options, rewards, terminals, is_initial_betas, log_pi, actions = rollout[i]
        #     options = tensor(options).unsqueeze(1).long()
        #     prev_options = tensor(prev_options).unsqueeze(1).long()
        #     terminals = tensor(terminals).unsqueeze(1)
        #     rewards = tensor(rewards).unsqueeze(1)
        #     is_initial_betas = tensor(is_initial_betas).unsqueeze(1)
        #     returns = rewards + config.discount * terminals * returns
        #
        #     q_omg = q_options.gather(1, options)
        #     log_action_prob = log_pi.gather(1, actions.unsqueeze(1))
        #     entropy_loss = (log_pi.exp() * log_pi).sum(-1).unsqueeze(1)
        #
        #     q_prev_omg = q_options.gather(1, prev_options)
        #     v_prev_omg = torch.max(q_options, dim=1, keepdim=True)[0]
        #     advantage_omg = q_prev_omg - v_prev_omg
        #     advantage_omg.add_(config.termination_regularizer)
        #     betas = betas.gather(1, prev_options)
        #     betas = betas * (1 - is_initial_betas)
        #     processed_rollout[i] = [q_omg, returns, betas, advantage_omg.detach(), log_action_prob, entropy_loss]

        # q_omg, returns, beta_omg, advantage_omg, log_action_prob, entropy_loss = map(lambda x: torch.cat(x, dim=0),
        #                                                                              zip(*processed_rollout))
        # pi_loss = -log_action_prob * (returns - q_omg.detach()) + config.entropy_weight * entropy_loss
        # pi_loss = pi_loss.mean()
        # q_loss = 0.5 * (q_omg - returns).pow(2).mean()
        # beta_loss = (advantage_omg * beta_omg).mean()

