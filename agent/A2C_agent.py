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

class A2CAgent:
    def __init__(self, config):
        self.config = config
        self.learning_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.learning_network.parameters())
        self.task = config.task_fn()
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0

    def episode(self, deterministic=False):
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while not self.config.max_episode_length or steps < self.config.max_episode_length:
            prob = self.learning_network.predict(np.stack([state]), True)
            action = self.policy.sample(prob, deterministic=deterministic)
            next_state, reward, done, info = self.task.step(action)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            total_reward += np.sum(reward * self.config.reward_weight)
            steps += 1
            state = next_state
            if done:
                break
            if not deterministic and self.total_steps > self.config.min_memory_size:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                prob, log_prob, value = self.learning_network.predict(states, False)
                _, _, v_next = self.learning_network.predict(next_states, False)
                terminals = self.learning_network.to_torch_variable(terminals).unsqueeze(1)
                rewards = self.learning_network.to_torch_variable(rewards).unsqueeze(1)
                actions = self.learning_network.to_torch_variable(actions, 'int64').unsqueeze(1)
                target = rewards + self.config.discount * v_next * (1 - terminals)
                target = target.detach()
                advantage = target - value
                value_loss = 0.5 * advantage.pow(2).mean()
                policy_loss = -(log_prob.gather(1, actions) * Variable(advantage.data)).mean()
                kl_loss = (prob * log_prob).sum(1).mean()

                self.optimizer.zero_grad()
                (value_loss + policy_loss + self.config.entropy_weight * kl_loss).backward()
                torch.nn.utils.clip_grad_norm(self.learning_network.parameters(), self.config.gradient_clip)
                self.optimizer.step()

        return total_reward, steps

    def run(self):
        window_size = 100
        ep = 0
        rewards = []
        steps = []
        avg_test_rewards = []
        while True:
            ep += 1
            reward, step = self.episode()
            rewards.append(reward)
            steps.append(step)
            avg_reward = np.mean(rewards[-window_size:])
            self.config.logger.info('episode %d, reward %f, avg reward %f, total steps %d, episode step %d' % (
                ep, reward, avg_reward, self.total_steps, step))

            if self.config.episode_limit and ep > self.config.episode_limit:
                return rewards, steps, avg_test_rewards

            if self.config.test_interval and ep % self.config.test_interval == 0:
                self.config.logger.info('Testing...')
                with open('data/%s-dqn-model-%s.bin' % (self.config.tag, self.task.name), 'wb') as f:
                    pickle.dump(self.learning_network.state_dict(), f)
                test_rewards = []
                for _ in range(self.config.test_repetitions):
                    reward, step = self.episode(True)
                    test_rewards.append(reward)
                avg_reward = np.mean(test_rewards)
                avg_test_rewards.append(avg_reward)
                self.config.logger.info('Avg reward %f(%f)' % (
                    avg_reward, np.std(test_rewards) / np.sqrt(self.config.test_repetitions)))
                with open('data/%sdqn-statistics-%s.bin' % (self.config.tag, self.task.name), 'wb') as f:
                    pickle.dump({'rewards': rewards,
                                 'test_rewards': avg_test_rewards}, f)
                if avg_reward > self.task.success_threshold:
                    break
