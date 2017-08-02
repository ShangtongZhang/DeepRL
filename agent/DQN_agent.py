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

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.learning_network = config.network_fn(config.optimizer_fn)
        self.target_network = config.network_fn(config.optimizer_fn)
        self.target_network.load_state_dict(self.learning_network.state_dict())
        self.task = config.task_fn()
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.history_buffer = None

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        if self.history_buffer is None:
            self.history_buffer = [np.zeros_like(state)] * self.config.history_length
        else:
            self.history_buffer.pop(0)
            self.history_buffer.append(state)
        state = np.vstack(self.history_buffer)
        total_reward = 0.0
        steps = 0
        while not self.config.max_episode_length or steps < self.config.max_episode_length:
            value = self.learning_network.predict(np.stack([self.task.normalize_state(state)]), True)
            if deterministic:
                action = np.argmax(value.flatten())
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value.flatten()))
            else:
                action = self.policy.sample(value.flatten())
            next_state, reward, done, info = self.task.step(action)
            self.history_buffer.pop(0)
            self.history_buffer.append(next_state)
            next_state = np.vstack(self.history_buffer)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            total_reward += reward
            steps += 1
            state = next_state
            if done:
                break
            if not deterministic and self.total_steps > self.config.exploration_steps:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.task.normalize_state(states)
                next_states = self.task.normalize_state(next_states)
                q_next = self.target_network.predict(next_states).detach()
                if self.config.double_q:
                    _, best_actions = self.learning_network.predict(next_states).detach().max(1)
                    q_next = q_next.gather(1, best_actions)
                else:
                    q_next, _ = q_next.max(1)
                terminals = self.learning_network.to_torch_variable(terminals).unsqueeze(1)
                rewards = self.learning_network.to_torch_variable(rewards).unsqueeze(1)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = self.learning_network.to_torch_variable(actions, 'int64').unsqueeze(1)
                q = self.learning_network.predict(states)
                q = q.gather(1, actions)
                loss = self.learning_network.criterion(q, q_next)
                self.learning_network.zero_grad()
                loss.backward()
                self.learning_network.optimizer.step()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()
        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward

    def run(self):
        window_size = 100
        ep = 0
        rewards = []
        avg_test_rewards = []
        while True:
            ep += 1
            reward = self.episode()
            rewards.append(reward)
            avg_reward = np.mean(rewards[-window_size:])
            self.config.logger.info('episode %d, epsilon %f, reward %f, avg reward %f, total steps %d' % (
                ep, self.policy.epsilon, reward, avg_reward, self.total_steps))

            if self.config.test_interval and ep % self.config.test_interval == 0:
                self.config.logger.info('Testing...')
                with open('data/%s-dqn-model-%s.bin' % (self.config.tag, self.task.name), 'wb') as f:
                    pickle.dump(self.learning_network.state_dict(), f)
                test_rewards = []
                for _ in range(self.config.test_repetitions):
                    test_rewards.append(self.episode(True))
                avg_reward = np.mean(test_rewards)
                avg_test_rewards.append(avg_reward)
                self.config.logger.info('Avg reward %f(%f)' % (
                    avg_reward, np.std(test_rewards) / np.sqrt(self.config.test_repetitions)))
                with open('data/%sdqn-statistics-%s.bin' % (self.config.tag, self.task.name), 'wb') as f:
                    pickle.dump({'rewards': rewards,
                                 'test_rewards': avg_test_rewards}, f)
                if avg_reward > self.task.success_threshold:
                    break