#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class ASquaredCA2CAgent(BaseAgent):
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

    def compute_adv(self, storage, mdp):
        config = self.config

        v = storage.__getattribute__('v_%s' % (mdp))
        adv = storage.__getattribute__('adv_%s' % (mdp))
        all_ret = storage.__getattribute__('ret_%s' % (mdp))

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

    def learn(self, storage, mdp, freeze_v=False):
        config = self.config
        log_prob, v, ret, adv, pi_hat = storage.cat(['log_pi_%s' % (mdp), 'v_%s' % (mdp), 'ret_%s' % (mdp), 'adv_%s' % (mdp), 'pi_hat'])

        if mdp == 'hat':
            entropy = -(pi_hat * pi_hat.add(1e-5).log()).sum(-1).mean()
        elif mdp == 'bar':
            entropy = 0
        else:
            raise NotImplementedError

        policy_loss = -(log_prob * adv.detach()).mean() - config.entropy_weight * entropy
        value_loss = (v - ret.detach()).pow(2).mul(0.5).mean()

        if freeze_v:
            value_loss = 0

        loss = policy_loss + value_loss
        return loss

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['adv_bar', 'adv_hat', 'ret_bar', 'ret_hat'])
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
            dist = torch.distributions.Categorical(probs=pi_hat)
            options = dist.sample()

            mean = prediction['mean'][self.worker_index, options]
            std = prediction['std'][self.worker_index, options]
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()

            pi_bar = self.compute_pi_bar(options.unsqueeze(-1), actions,
                                         prediction['mean'], prediction['std'])

            v_bar = prediction['q_o'].gather(1, options.unsqueeze(-1))
            v_hat = (prediction['q_o'] * pi_hat).sum(-1).unsqueeze(-1)

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
                         'pi_hat': pi_hat,
                         'log_pi_hat': pi_hat[self.worker_index, options].add(1e-5).log().unsqueeze(-1),
                         'log_pi_bar': pi_bar.add(1e-5).log(),
                         'v_bar': v_bar,
                         'v_hat': v_hat})

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options

            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
        dist = torch.distributions.Categorical(pi_hat)
        options = dist.sample()
        v_bar = prediction['q_o'].gather(1, options.unsqueeze(-1))
        v_hat = (prediction['q_o'] * pi_hat).sum(-1).unsqueeze(-1)

        storage.add(prediction)
        storage.add({
            'v_bar': v_bar,
            'v_hat': v_hat,
        })
        storage.placeholder()

        self.compute_adv(storage, 'bar')
        self.compute_adv(storage, 'hat')

        if config.learning == 'all':
            mdps = ['hat', 'bar']
            np.random.shuffle(mdps)
            loss1 = self.learn(storage, mdps[0])
            loss2 = self.learn(storage, mdps[1], freeze_v=config.freeze_v)
            loss = loss1 + loss2
        elif config.learning == 'alt':
            if self.count % 2:
                loss = self.learn(storage, 'hat')
            else:
                loss = self.learn(storage, 'bar')
            self.count += 1
        else:
            raise NotImplementedError

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.opt.step()
