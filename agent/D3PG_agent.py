#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *
from component import *
from utils import *
import numpy as np
import time
import os
import pickle
import torch
from .BaseAgent import *

class D3PGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()
        self.states = self.config.state_normalizer(self.states)
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.random_process = config.random_process_fn(self.task.action_dim)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            actions = self.network.actor(states, True)
            actions += self.random_process.sample()
            next_states, rewards, terminals, _ = self.task.step(actions)
            next_states = self.config.state_normalizer(next_states)
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([states, actions, rewards, 1 - terminals, next_states])
            states = next_states
            self.total_steps += config.num_workers

        self.states = states

        _, q_next, best = self.target_network.actor(states)
        returns = q_next.max(1)[0].unsqueeze(1).detach()
        for i in reversed(range(len(rollout))):
            states, actions, rewards, terminals, next_states = rollout[i]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns

            q = self.network.critic(states, actions)
            q_loss = (q - returns).pow(2).mul(0.5).mean() * config.value_loss_weight

            self.optimizer.zero_grad()
            q_loss.backward()
            self.optimizer.step()

            _, q_values, _ = self.network.actor(states)
            policy_loss = -q_values.sum(1).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.network.zero_critic_grad()
            self.optimizer.step()
            self.soft_update(self.target_network, self.network)

        self.evaluate(config.rollout_length)