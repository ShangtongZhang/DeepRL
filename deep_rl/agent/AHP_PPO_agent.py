#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class AHPPPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0

        self.worker_index = tensor(np.arange(config.num_workers)).long()
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = tensor(np.zeros(config.num_workers)).long()

        self.count = 0

    def sample_stop(self, beta):
        stop = tensor(np.random.rand(*beta.size())) < beta
        return stop

    def compute_prob_a(self, prev_o, stop, beta, o, pi_o, pi_a):
        p_sp = torch.where(stop, beta, 1 - beta)
        p_op = torch.where(stop, pi_o, (prev_o == o).float().detach())
        return p_sp * p_op * pi_a

    def compute_v(self, u_o, prev_option, is_initial_states):
        v_init = u_o[:, [-1]]
        v = u_o.gather(1, prev_option)
        v = torch.where(is_initial_states, v_init, v)
        return v

    def compute_adv(self, storage):
        config = self.config
        ret = storage.v[-1].detach()
        advantages = tensor(np.zeros((config.num_workers, 1)))
        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            if not config.use_gae:
                advantages = ret - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = ret.detach()

    def learn(self, storage):
        config = self.config
        states, actions, options, prob_a, returns, advantages, prev_options, inits, mean, std, stop = \
            storage.cat(['s', 'a', 'o', 'prob_a', 'ret', 'adv', 'prev_o', 'init', 'mean', 'std', 'stop'])
        log_probs_old = prob_a.add(1e-5).log().detach()
        advantages = (advantages - advantages.mean()) / advantages.std()
        self.logger.add_histogram('adv', advantages, log_level=5)

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()

                sampled_states = states[batch_indices]
                sampled_prev_o = prev_options[batch_indices]
                sampled_init = inits[batch_indices]
                sampled_stop = stop[batch_indices]

                sampled_options = options[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states)
                dist = torch.distributions.Normal(
                    prediction['mean'][range(sampled_options.size(0)), sampled_options.view(-1)],
                    prediction['std'][range(sampled_options.size(0)), sampled_options.view(-1)])
                pi_a = dist.log_prob(sampled_actions).sum(-1).exp().unsqueeze(-1)
                pi_a = self.compute_prob_a(sampled_prev_o,
                                           sampled_stop,
                                           prediction['beta'].gather(1, sampled_options),
                                           sampled_options,
                                           prediction['inter_pi'].gather(1, sampled_options),
                                           pi_a)
                log_pi_a = pi_a.add(1e-5).log()
                v = self.compute_v(prediction['u_o'], sampled_prev_o, sampled_init)

                ratio = (log_pi_a - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean()

                value_loss = 0.5 * (sampled_returns - v).pow(2).mean()
                self.logger.add_scalar('v_loss', value_loss.item(), log_level=5)

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            beta = prediction['beta'][self.worker_index, self.prev_options]
            stop = self.sample_stop(beta)
            stop = torch.where(self.is_initial_states, tensor(np.ones(stop.size())).byte(), stop)

            dist = torch.distributions.Categorical(probs=prediction['inter_pi'])
            options = dist.sample()
            options = torch.where(stop, options, self.prev_options)

            self.logger.add_scalar('beta', prediction['beta'][self.worker_index, self.prev_options], log_level=5)
            self.logger.add_scalar('option', options[0], log_level=5)
            self.logger.add_scalar('pi_hat_ent', dist.entropy(), log_level=5)
            self.logger.add_scalar('pi_hat_o', dist.log_prob(options).exp(), log_level=5)

            mean = prediction['mean'][self.worker_index, options]
            std = prediction['std'][self.worker_index, options]
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            pi_a = dist.log_prob(actions).sum(-1).exp().unsqueeze(-1)

            prob_a = self.compute_prob_a(self.prev_options.unsqueeze(-1),
                                         stop.unsqueeze(-1),
                                         beta.unsqueeze(-1),
                                         options.unsqueeze(-1),
                                         prediction['inter_pi'][self.worker_index, options].unsqueeze(-1),
                                         pi_a)

            v = self.compute_v(prediction['u_o'], self.prev_options.unsqueeze(-1),
                               self.is_initial_states.unsqueeze(-1))

            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)

            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'a': actions,
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         's': tensor(states),
                         'init': self.is_initial_states.unsqueeze(-1),
                         'prob_a': prob_a,
                         'stop': stop.unsqueeze(-1),
                         'v': v})

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options

            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        v = self.compute_v(prediction['u_o'], self.prev_options.unsqueeze(-1),
                           self.is_initial_states.unsqueeze(-1))
        storage.add(prediction)
        storage.add({
            'v': v
        })
        storage.placeholder()

        [o] = storage.cat(['o'])
        for i in range(config.num_o):
            self.logger.add_scalar('option_%d' % (i), (o == i).float().mean(), log_level=1)

        self.compute_adv(storage)
        self.learn(storage)
