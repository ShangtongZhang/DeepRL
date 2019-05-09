#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from skimage import color


class PPOCAgent(BaseAgent):
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

        self.all_options = []

    def compute_pi_hat(self, prediction, prev_option, is_intial_states):
        inter_pi = prediction['inter_pi']
        mask = torch.zeros_like(inter_pi)
        mask[self.worker_index, prev_option] = 1
        beta = prediction['beta']
        pi_hat = (1 - beta) * mask + beta * inter_pi
        is_intial_states = is_intial_states.view(-1, 1).expand(-1, inter_pi.size(1))
        pi_hat = torch.where(is_intial_states, inter_pi, pi_hat)
        return pi_hat

    def compute_pi_bar(self, options, action, mean, std):
        options = options.unsqueeze(-1).expand(-1, -1, mean.size(-1))
        mean = mean.gather(1, options).squeeze(1)
        std = std.gather(1, options).squeeze(1)
        dist = torch.distributions.Normal(mean, std)
        pi_bar = dist.log_prob(action).sum(-1).exp().unsqueeze(-1)
        return pi_bar

    def compute_log_pi_a(self, options, pi_hat, action, mean, std, mdp):
        if mdp == 'hat':
            return pi_hat.add(1e-5).log().gather(1, options)
        elif mdp == 'bar':
            pi_bar = self.compute_pi_bar(options, action, mean, std)
            return pi_bar.add(1e-5).log()
        else:
            raise NotImplementedError

    def compute_adv(self, storage):
        config = self.config

        v = storage.v
        adv = storage.adv
        all_ret = storage.ret

        ret = v[-1].detach()
        advantages = tensor(np.zeros((config.num_workers, 1)))
        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            if not config.use_gae:
                advantages = ret - v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * v[i + 1] - v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            adv[i] = advantages.detach()
            all_ret[i] = ret.detach()

    def learn(self, storage):
        config = self.config

        states, actions, log_pi_bar_old, options, returns, advantages, inits, prev_options = storage.cat(
            ['s', 'a', 'log_pi_bar', 'o', 'ret', 'adv', 'init', 'prev_o'])
        actions = actions.detach()
        log_pi_bar_old = log_pi_bar_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_options = options[batch_indices]
                sampled_log_pi_bar_old = log_pi_bar_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                sampled_inits = inits[batch_indices]
                sampled_prev_options = prev_options[batch_indices]

                prediction = self.network(sampled_states)
                pi_bar = self.compute_pi_bar(sampled_options, sampled_actions, prediction['mean'], prediction['std'])
                log_pi_bar = pi_bar.add(1e-5).log()
                ratio = (log_pi_bar - sampled_log_pi_bar_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean()

                beta_adv = prediction['q_o'].gather(1, sampled_prev_options) - \
                           (prediction['q_o'] * prediction['inter_pi']).sum(-1).unsqueeze(-1)
                beta_adv = beta_adv.detach() + config.beta_reg
                beta_loss = prediction['beta'].gather(1, sampled_prev_options) * (1 - sampled_inits).float() * beta_adv
                beta_loss = beta_loss.mean()

                q_loss = (prediction['q_o'].gather(1, sampled_options) - sampled_returns.detach()).pow(2).mul(0.5).mean()

                ent = -(prediction['log_inter_pi'] * prediction['inter_pi']).sum(-1).mean()
                inter_pi_loss = -(prediction['log_inter_pi'].gather(1, sampled_options) * sampled_advantages).mean()\
                                - config.entropy_weight * ent

                self.opt.zero_grad()
                (policy_loss + beta_loss + q_loss + inter_pi_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()


    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
            dist = torch.distributions.Categorical(probs=pi_hat)
            options = dist.sample()

            self.logger.add_scalar('beta', prediction['beta'][self.worker_index, self.prev_options], log_level=5)
            self.logger.add_scalar('option', options[0], log_level=5)
            self.logger.add_scalar('pi_hat_ent', dist.entropy(), log_level=5)
            self.logger.add_scalar('pi_hat_o', dist.log_prob(options).exp(), log_level=5)

            mean = prediction['mean'][self.worker_index, options]
            std = prediction['std'][self.worker_index, options]
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()

            pi_bar = self.compute_pi_bar(options.unsqueeze(-1), actions,
                                         prediction['mean'], prediction['std'])

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
                         'v': prediction['q_o'][self.worker_index, options].unsqueeze(-1),
                         'log_pi_bar': pi_bar.add(1e-5).log(),
                         })

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options

            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
        dist = torch.distributions.Categorical(pi_hat)
        options = dist.sample()

        storage.add(prediction)
        storage.add({
            'v': prediction['q_o'][self.worker_index, options].unsqueeze(-1)
        })
        storage.placeholder()

        self.compute_adv(storage)
        self.learn(storage)
