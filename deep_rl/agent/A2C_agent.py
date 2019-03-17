#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(config.state_normalizer(states))
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
