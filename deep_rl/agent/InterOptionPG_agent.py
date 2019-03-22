#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class InterOptionPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.worker_index = tensor(np.arange(config.num_workers)).long()

        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def compose_pi_hat(self, prediction, prev_option, is_intial_states):
        inter_pi = prediction['inter_pi']
        mask = torch.zeros_like(inter_pi)
        mask[:, prev_option] = 1
        beta = prediction['beta']
        self.logger.add_scalar('beta', beta[0, prev_option[0]])
        if self.config.beta_grad == 'direct':
            beta = beta.detach()
        pi_hat = (1 - beta) * mask + beta * inter_pi

        is_intial_states = is_intial_states.view(-1, 1).expand(-1 ,inter_pi.size(1))
        pi_hat = torch.where(is_intial_states, inter_pi, pi_hat)
        return pi_hat

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'inter_pi',
                                                  'log_inter_pi', 'pi_hat', 'ent_pi_hat', 'all_pi'])
        for _ in range(config.rollout_length):
            prediction = self.network(self.states)

            pi_hat = self.compose_pi_hat(prediction, self.prev_options, self.is_initial_states)
            dist = torch.distributions.Categorical(probs=pi_hat)
            options = dist.sample()
            ent_pi_hat = dist.entropy()

            all_pi = prediction['pi']
            prediction['pi'] = prediction['pi'][self.worker_index, options]
            prediction['log_pi'] = prediction['log_pi'][self.worker_index, options]
            dist = torch.distributions.Categorical(probs=prediction['pi'])
            actions = dist.sample()
            entropy = dist.entropy()

            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         'ent': entropy.unsqueeze(-1),
                         'a': actions.unsqueeze(-1),
                         'init': self.is_initial_states.unsqueeze(-1).float(),
                         'pi_hat': pi_hat,
                         'ent_pi_hat': ent_pi_hat.unsqueeze(-1),
                         'all_pi': all_pi,
                         })
            self.logger.add_scalar('pi_hat_ent', ent_pi_hat.mean())

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            prediction = self.target_network(self.states)
            storage.placeholder()
            betas = prediction['beta'][self.worker_index, self.prev_options]
            ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
                  betas * (prediction['q'] * prediction['inter_pi']).mean(-1)
            ret = ret.unsqueeze(-1)

        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            adv = ret - storage.q[i].gather(1, storage.o[i])
            storage.ret[i] = ret
            storage.adv[i] = adv

            v = (storage.q[i] * storage.inter_pi[i]).mean(-1).unsqueeze(-1)
            q = storage.q[i].gather(1, storage.prev_o[i])
            storage.beta_adv[i] = q - v + config.beta_reg

        q, beta, log_pi, ret, adv, beta_adv, ent, option, action, initial_states, prev_o, pi_hat, ent_pi_hat, all_pi = \
            storage.cat(['q', 'beta', 'log_pi', 'ret', 'adv', 'beta_adv', 'ent', 'o', 'a', 'init', 'prev_o', 'pi_hat', 'ent_pi_hat', 'all_pi'])

        q_o = q.gather(1, option)
        v_hat = (q * pi_hat).mean(-1).unsqueeze(-1)
        adv_hat = (q_o - v_hat).detach()
        if config.pi_hat_grad == 'sample':
            pi_hat_loss = -pi_hat.add(1e-5).log().gather(1, option) * adv_hat - config.ent_hat * ent_pi_hat
        elif config.pi_hat_grad == 'expected':
            pi_hat_loss = -(pi_hat * q.detach()).sum(-1) - config.ent_hat * ent_pi_hat
        elif config.pi_hat_grad == 'posterior':
            pi_a = all_pi.gather(-1, action.unsqueeze(-1).expand(-1, pi_hat.size(1), -1))
            post = pi_hat * pi_a.squeeze(-1)
            post = post / post.sum(-1).unsqueeze(-1)
            post = post.detach()
            pi_hat_loss = -(pi_hat.add(1e-5).log() * q.detach() * post).sum(-1) - config.ent_hat * ent_pi_hat
        else:
            raise NotImplementedError
        pi_hat_loss = pi_hat_loss.mean()

        q_loss = (q_o - ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(log_pi.gather(1, action) * adv.detach()) - config.entropy_weight * ent
        pi_loss = pi_loss.mean()
        if config.beta_grad == 'direct':
            # self.logger.add_histogram('beta_adv', beta_adv)
            beta_loss = (beta.gather(1, prev_o) * beta_adv.detach() * (1 - initial_states)).mean()
        elif config.beta_grad == 'indirect':
            beta_loss = 0
        else:
            raise NotImplementedError

        self.optimizer.zero_grad()
        (pi_hat_loss + pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
