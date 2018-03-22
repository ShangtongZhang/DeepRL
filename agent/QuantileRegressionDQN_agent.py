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

class QuantileRegressionDQNAgent:
    def __init__(self, config):
        self.config = config
        self.learning_network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.learning_network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.learning_network.state_dict())
        self.task = config.task_fn()
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = self.learning_network.tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))

    def huber(self, x):
        cond = (x < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while True:
            value = self.learning_network.predict(np.stack([self.task.normalize_state(state)])).squeeze(0).data
            value = (value * self.quantile_weight).sum(-1).cpu().numpy().flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, _ = self.task.step(action)
            total_reward += reward
            reward = self.config.reward_shift_fn(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1
            state = next_state
            if done:
                break
            if not deterministic and self.total_steps > self.config.exploration_steps:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.task.normalize_state(states)
                next_states = self.task.normalize_state(next_states)

                quantiles_next = self.target_network.predict(next_states).data
                q_next = (quantiles_next * self.quantile_weight).sum(-1)
                _, a_next = torch.max(q_next, dim=1)
                a_next = a_next.view(-1, 1, 1).expand(-1, -1, quantiles_next.size(2))
                quantiles_next = quantiles_next.gather(1, a_next).squeeze(1)

                rewards = self.learning_network.tensor(rewards)
                terminals = self.learning_network.tensor(terminals)
                quantiles_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * quantiles_next

                quantiles = self.learning_network.predict(states)
                actions = self.learning_network.tensor(actions, torch.LongTensor)
                actions = actions.view(-1, 1, 1).expand(-1, -1, quantiles.size(2))
                quantiles = quantiles.gather(1, Variable(actions)).squeeze(1)

                loss = 0.0
                for i in range(self.config.num_quantiles):
                    diff = Variable(quantiles_next[:, i].contiguous().view(-1, 1)) - quantiles
                    loss += self.huber(diff) * Variable(self.cumulative_density.view(1, -1) - (diff.data < 0).float()).abs()

                self.optimizer.zero_grad()
                loss.sum(-1).mean().backward()
                self.optimizer.step()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()
        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self.learning_network.state_dict(), f)

    def close(self):
        pass
