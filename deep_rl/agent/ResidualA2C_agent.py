#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class ResidualA2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.target_network.load_state_dict(self.network.state_dict())

        self.episode_rewards = []
        self.online_rewards = np.zeros(config.num_workers)

        self.worker_index = tensor(np.ones(config.num_workers)).long()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['d', 'rd'])
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, _ = self.task.step(to_np(prediction['a']))
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)})

            states = next_states
            self.total_steps += config.num_workers
            if self.total_steps / config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states
        prediction = self.network(config.state_normalizer(states))
        storage.add(prediction)
        storage.add({
            's': tensor(states),
        })
        storage.placeholder()

        returns = prediction['q_a'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            storage.ret[i] = returns.detach()

            if config.multi_step:
                d_loss = (storage.q_a[i] - returns).pow(2).mul(0.5)
                storage.d[i] = d_loss
            elif not config.symmetric:
                if config.target_net_residual:
                    target_net = self.target_network
                else:
                    target_net = self.network

                with torch.no_grad():
                    prediction = target_net(storage.s[i + 1])
                    q_next = prediction['q'].gather(1, storage.a[i + 1].unsqueeze(1))
                    target = storage.r[i] + config.discount * storage.m[i] * q_next
                d_loss = (storage.q_a[i] - target).pow(2).mul(0.5)

                prediction = self.network(storage.s[i + 1])
                q_next = prediction['q_a']
                with torch.no_grad():
                    q_next_hat = prediction['q'].gather(1, storage.a[i + 1].unsqueeze(1))
                    q = target_net(storage.s[i], storage.a[i])['q_a']
                    td_error = storage.r[i] + config.discount * storage.m[i] * q_next_hat - q
                rd_loss = config.discount * storage.m[i] * td_error * q_next

                storage.d[i] = d_loss
                storage.rd[i] = rd_loss
            else:
                raise NotImplementedError

        pi, q, entropy, d_loss = storage.cat(['pi', 'q', 'ent', 'd'])
        if config.multi_step:
            rd_loss = 0
        else:
            [rd_loss] = storage.cat(['rd'])

        # log_pi_a, adv, v = storage.cat(['log_pi_a', 'adv', 'v'])
        # policy_loss = -(log_pi_a * adv.detach()).mean()

        policy_loss = -(pi * q.detach()).sum(-1).mean()
        entropy_loss = entropy.mean()
        value_loss = (d_loss + config.residual * rd_loss).mean()

        config.logger.add_scalar('v_loss', value_loss.item())
        config.logger.add_scalar('pi_loss', policy_loss.item())
        config.logger.add_scalar('pi_ent', entropy_loss.item())

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
