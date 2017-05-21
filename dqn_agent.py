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

    def get_state(self, history_buffer):
        if self.history_length > 1:
            return np.vstack(history_buffer)
        return history_buffer[0]

    def episode(self):
        episode_start_time = time.time()
        state = self.task.reset()
        history_buffer = [state] * self.history_length
        total_reward = 0.0
        steps = 0
        while not self.step_limit or steps < self.step_limit:
            state = self.get_state(history_buffer)
            value = self.learning_network.predict(np.reshape(state, (1, ) + state.shape))
            action = self.policy.sample(value.flatten())
            next_state, reward, done, info = self.task.step(action)
            history_buffer.pop(0)
            history_buffer.append(next_state)
            next_state = self.get_state(history_buffer)
            total_reward += reward
            self.replay.feed([state, action, reward, next_state, int(done)])
            steps += 1
            self.total_steps += 1
            if done:
                break
            if self.total_steps > self.explore_steps:
                sample_start_time = time.time()
                experiences = self.replay.sample()
                self.logger.debug('sample time %f' % (time.time() - sample_start_time))
                states, actions, rewards, next_states, terminals = experiences
                predict_start_time = time.time()
                targets = self.learning_network.predict(states)
                q_next = self.target_network.predict(next_states)
                self.logger.debug('prediction time %f' % (time.time() - predict_start_time))
                q_next = np.max(q_next, axis=1)
                q_next = np.where(terminals, 0, q_next)
                q_next = rewards + self.discount * q_next
                targets[np.arange(len(actions)), actions] = q_next
                minibatch_start_time = time.time()
                self.learning_network.learn(states, targets)
                self.logger.debug('minibatch time %f' % (time.time() - minibatch_start_time))
            if self.total_steps % self.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())
            if self.total_steps > self.explore_steps:
                self.policy.update_epsilon()
        episode_time = time.time() - episode_start_time
        info = self.process.memory_full_info()
        if hasattr(info, 'swap'):
            info_stat = info.swap
        elif hasattr(info, 'pfaults'):
            info_stat = info.pfaults
        else:
            info_stat = -1
        self.logger.debug('episode steps %d, episode time %f, time per step %f, memory_info %d' %
                          (steps, episode_time, episode_time / float(steps), info_stat))
        return total_reward

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
            if avg_reward > self.task.success_threshold:
                break
