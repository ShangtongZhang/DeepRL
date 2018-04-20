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

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while True:
            value = self.network.predict(np.stack([self.config.state_normalizer(state)]), True).flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, _ = self.task.step(action)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1
            state = next_state

            if not deterministic and self.total_steps > self.config.exploration_steps:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                q_next = self.target_network.predict(next_states, False).detach()
                if self.config.double_q:
                    _, best_actions = self.network.predict(next_states).detach().max(1)
                    q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    q_next, _ = q_next.max(1)
                terminals = self.network.variable(terminals)
                rewards = self.network.variable(rewards)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = self.network.variable(actions, torch.LongTensor).unsqueeze(1)
                q = self.network.predict(states, False)
                q = q.gather(1, actions).squeeze(1)
                loss = self.criterion(q, q_next)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps
