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

class DQNAgent:
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

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        self.history_buffer = [state] * self.config.history_length
        state = np.vstack(self.history_buffer)
        total_reward = 0.0
        steps = 0
        while True:
            value = self.learning_network.predict(np.stack([self.task.normalize_state(state)]), True).flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, info = self.task.step(action)
            done = (done or (self.config.max_episode_length and steps > self.config.max_episode_length))
            self.history_buffer.pop(0)
            self.history_buffer.append(next_state)
            next_state = np.vstack(self.history_buffer)
            total_reward += np.sum(reward * self.config.reward_weight)
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
                if self.config.hybrid_reward:
                    q_next = self.target_network.predict(next_states, True)
                    target = []
                    for q_next_ in q_next:
                        if self.config.target_type == self.config.q_target:
                            target.append(q_next_.detach().max(1)[0])
                        elif self.config.target_type == self.config.expected_sarsa_target:
                            target.append(q_next_.detach().mean(1))
                    target = torch.stack(target, dim=1).detach()
                    terminals = self.learning_network.to_torch_variable(terminals).unsqueeze(1)
                    rewards = self.learning_network.to_torch_variable(rewards)
                    target = self.config.discount * target * (1 - terminals)
                    target.add_(rewards)
                    q = self.learning_network.predict(states, True)
                    q_action = []
                    actions = self.learning_network.to_torch_variable(actions, 'int64').unsqueeze(1)
                    for q_ in q:
                        q_action.append(q_.gather(1, actions))
                    q_action = torch.cat(q_action, dim=1)
                    loss = self.learning_network.criterion(q_action, target)
                else:
                    q_next = self.target_network.predict(next_states, False).detach()
                    if self.config.double_q:
                        _, best_actions = self.learning_network.predict(next_states).detach().max(1)
                        q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    else:
                        q_next, _ = q_next.max(1)
                    terminals = self.learning_network.to_torch_variable(terminals)
                    rewards = self.learning_network.to_torch_variable(rewards)
                    q_next = self.config.discount * q_next * (1 - terminals)
                    q_next.add_(rewards)
                    actions = self.learning_network.to_torch_variable(actions, 'int64').unsqueeze(1)
                    q = self.learning_network.predict(states, False)
                    q = q.gather(1, actions).squeeze(1)
                    loss = self.criterion(q, q_next)
                self.optimizer.zero_grad()
                loss.backward()
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
