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
        self.report_interval = 1000

    def get_state(self, history_buffer):
        return np.vstack(history_buffer)

    def episode(self):
        episode_start_time = time.time()
        state = self.task.reset()
        history_buffer = [state] * self.history_length
        total_reward = 0.0
        steps = 0
        while not self.step_limit or steps < self.step_limit:
            state = self.get_state(history_buffer)
            state = self.task.normalize_state(state)
            value = self.learning_network.predict(np.reshape(state, (1, ) + state.shape))
            action = self.policy.sample(value.flatten())
            next_state, reward, done, info = self.task.step(action)
            history_buffer.pop(0)
            history_buffer.append(next_state)
            next_state = self.get_state(history_buffer)
            self.replay.feed([state, action, reward, next_state, int(done)])
            total_reward += reward
            steps += 1
            self.total_steps += 1
            if done:
                break
            if self.total_steps > self.explore_steps:
                sample_start_time = time.time()
                experiences = self.replay.sample()
                if self.total_steps % self.report_interval == 0:
                    self.logger.debug('sample time %f' % (time.time() - sample_start_time))
                states, actions, rewards, next_states, terminals = experiences
                states = self.task.normalize_state(states)
                next_states = self.task.normalize_state(next_states)
                predict_start_time = time.time()
                q_next = self.target_network.predict(next_states)
                if self.total_steps % self.report_interval == 0:
                    self.logger.debug('prediction time %f' % (time.time() - predict_start_time))
                q_next = np.max(q_next, axis=1)
                q_next = np.where(terminals, 0, q_next)
                q_next = rewards + self.discount * q_next
                minibatch_start_time = time.time()
                self.learning_network.learn(states, actions, q_next)
                # self.learning_network.clippedLearn(states, actions, q_next)
                if self.total_steps % self.report_interval == 0:
                    self.logger.debug('minibatch time %f' % (time.time() - minibatch_start_time))
            if self.total_steps % self.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())
            if self.total_steps > self.explore_steps:
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
            if ep % 1000 == 0:
                self.save('data/dqn-episode-%d.bin' % (ep))
            rewards.append(reward)
            avg_reward = np.mean(rewards[-window_size:])
            self.logger.info('episode %d, epsilon %f, reward %f, avg reward %f, total steps %d' % (
                ep, self.policy.epsilon, reward, avg_reward, self.total_steps))
            if avg_reward > self.task.success_threshold:
                break
