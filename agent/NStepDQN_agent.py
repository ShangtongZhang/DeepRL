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

class NStepDQNAgent:
    def __init__(self, config):
        self.config = config
        self.learning_network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.learning_network.parameters())
        self.target_network.load_state_dict(self.learning_network.state_dict())
        self.task = config.task_fn()
        self.policy = config.policy_fn()

        self.total_steps = 0
        self.states = self.task.reset()
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

    def close(self):
        self.task.close()

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self.learning_network.state_dict(), f)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for i in range(config.rollout_length):
            q = self.learning_network.predict(states)
            actions = [self.policy.sample(v) for v in q.data.cpu().numpy()]
            actions = config.action_shift_fn(actions)
            next_states, rewards, terminals, _ = self.task.step(actions)
            self.episode_rewards += rewards
            rewards = config.reward_shift_fn(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    next_states[i] = self.task.reset(i)
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([q, actions, rewards, 1 - terminals])
            states = next_states

            self.policy.update_epsilon()
            self.total_steps += config.num_workers
            if self.total_steps / config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())

        self.states = states

        processed_rollout = [None] * (len(rollout))
        returns = self.target_network.predict(states).data
        returns, _ = torch.max(returns, dim=1, keepdim=True)
        for i in reversed(range(len(rollout))):
            q, actions, rewards, terminals = rollout[i]
            actions = self.learning_network.tensor(actions, torch.LongTensor).unsqueeze(1)
            q = q.gather(1, Variable(actions))
            terminals = self.learning_network.tensor(terminals).unsqueeze(1)
            rewards = self.learning_network.tensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns
            processed_rollout[i] = [q, returns]

        q, returns= map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        loss = 0.5 * (q - Variable(returns)).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()