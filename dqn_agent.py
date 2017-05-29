#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *
from replay import *
from policy import *
import numpy as np
import time
import psutil
import os
import pickle

class DQNAgent:
    def __init__(self,
                 task_fn,
                 network_fn,
                 optimizer_fn,
                 policy_fn,
                 replay_fn,
                 discount,
                 step_limit,
                 target_network_update_freq,
                 explore_steps,
                 history_length,
                 test_interval,
                 test_repetitions,
                 logger):
        self.learning_network = network_fn(optimizer_fn)
        self.target_network = network_fn(optimizer_fn)
        self.target_network.load_state_dict(self.learning_network.state_dict())
        self.task = task_fn()
        self.step_limit = step_limit
        self.replay = replay_fn()
        self.discount = discount
        self.target_network_update_freq = target_network_update_freq
        self.policy = policy_fn()
        self.total_steps = 0
        self.explore_steps = explore_steps
        self.history_length = history_length
        self.logger = logger
        self.process = psutil.Process(os.getpid())
        self.test_interval = test_interval
        self.test_repetitions = test_repetitions

    def get_state(self, history_buffer):
        return np.vstack(history_buffer)

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        history_buffer = [state] * self.history_length
        total_reward = 0.0
        steps = 0
        while not self.step_limit or steps < self.step_limit:
            state = self.get_state(history_buffer)
            state = self.task.normalize_state(state)
            value = self.learning_network.predict(np.reshape(state, (1, ) + state.shape))
            if deterministic:
                action = np.argmax(value.flatten())
            else:
                action = self.policy.sample(value.flatten())
            next_state, reward, done, info = self.task.step(action)
            history_buffer.pop(0)
            history_buffer.append(next_state)
            if not deterministic:
                next_state = self.get_state(history_buffer)
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            total_reward += reward
            steps += 1
            if done:
                break
            if not deterministic and self.total_steps > self.explore_steps:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.task.normalize_state(states)
                next_states = self.task.normalize_state(next_states)
                q_next = self.target_network.predict(next_states)
                q_next = np.max(q_next, axis=1)
                q_next = np.where(terminals, 0, q_next)
                q_next = rewards + self.discount * q_next
                self.learning_network.learn(states, actions, q_next)
            if not deterministic and self.total_steps % self.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())
            if not deterministic and self.total_steps > self.explore_steps:
                self.policy.update_epsilon()
        episode_time = time.time() - episode_start_time
        info = self.process.memory_info()
        self.logger.debug('episode steps %d, episode time %f, time per step %f, rss %d, vms %d' %
                          (steps, episode_time, episode_time / float(steps), info.rss, info.vms))
        return total_reward

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.learning_network.state_dict(), f)

    def run(self):
        window_size = 100
        ep = 0
        rewards = []
        while True:
            ep += 1
            reward = self.episode()
            rewards.append(reward)
            avg_reward = np.mean(rewards[-window_size:])
            self.logger.info('episode %d, epsilon %f, reward %f, avg reward %f, total steps %d' % (
                ep, self.policy.epsilon, reward, avg_reward, self.total_steps))

            if ep % self.test_interval == 0:
                self.logger.info('Testing...')
                self.save('data/dqn-episode-%d.bin' % (ep))
                test_rewards = []
                for _ in range(self.test_repetitions):
                    test_rewards.append(self.episode(True))
                avg_reward = np.mean(test_rewards)
                self.logger.info('Avg reward %f(%f)' % (
                    avg_reward, np.std(test_rewards) / np.sqrt(self.test_repetitions)))
                if avg_reward > self.task.success_threshold:
                    break