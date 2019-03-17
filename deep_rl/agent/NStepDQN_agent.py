#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *


class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        for _ in range(config.rollout_length):
            q = self.network(self.config.state_normalizer(states))

            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q))

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            storage.add({'q': q,
                         'a': tensor(actions).unsqueeze(-1).long(),
                         'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        storage.placeholder()

        ret = self.target_network(config.state_normalizer(states)).detach()
        ret = torch.max(ret, dim=1, keepdim=True)[0]
        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            storage.ret[i] = ret

        q, action, ret = storage.cat(['q', 'a', 'ret'])
        loss = 0.5 * (q.gather(1, action) - ret).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
