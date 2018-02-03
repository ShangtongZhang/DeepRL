#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch.multiprocessing as mp
from network import *
from utils import *
from component import *
import pickle
import os
import time
import gym.monitoring

class A2CAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.evaluator = self.task.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.states = self.task.reset()

        self.episode_counts = np.zeros(config.num_workers)
        self.episode_rewards = np.zeros(config.num_workers)
        self.total_rewards = np.zeros(config.num_workers)
        self.prev_episode_counts = 0.0
        self.prev_total_rewards = 0.0

    def close(self):
        self.task.close()

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self.network.state_dict(), f)

    def evaluate(self):
        state = self.evaluator.reset()
        total_rewards = 0
        steps = 0
        while True:
            prob, _, _ = self.network.predict(np.stack([state]))
            action = self.policy.sample(prob.data.cpu().numpy().flatten(), True)
            state, reward, done, _ = self.evaluator.step(action)
            total_rewards += reward
            steps += 1
            if done:
                break
        return total_rewards, steps

    def episode(self, deterministic=False):
        config = self.config
        for _ in range(config.iteration_log_interval):
            self.iteration(deterministic)
        new_episode_counts = np.sum(self.episode_counts)
        new_total_rewards = np.sum(self.total_rewards)
        avg_reward = (new_total_rewards - self.prev_total_rewards) / \
                     (new_episode_counts - self.prev_episode_counts + 1e-5)
        self.prev_total_rewards = new_total_rewards
        self.prev_episode_counts = new_episode_counts
        return avg_reward, config.rollout_length * config.num_workers * \
               config.iteration_log_interval

    def iteration(self, deterministic=False):
        if deterministic:
            return self.evaluate()

        config = self.config
        rollout = []
        states = self.states
        for i in range(config.rollout_length):
            prob, log_prob, value = self.network.predict(states)
            actions = [self.policy.sample(p, deterministic) for p in prob.data.cpu().numpy()]
            actions = config.action_shift_fn(actions)
            next_states, rewards, terminals, _ = self.task.step(actions)
            self.episode_rewards += rewards
            rewards = config.reward_shift_fn(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    next_states[i] = self.task.reset(i)
                    self.episode_counts[i] += 1
                    self.total_rewards[i] += self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([prob, log_prob, value, actions, rewards, 1 - terminals])
            states = next_states

        self.states = states
        _, _, pending_value = self.network.predict(states)
        rollout.append([None, None, pending_value, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = self.network.FloatTensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.data
        for i in reversed(range(len(rollout) - 1)):
            prob, log_prob, value, actions, rewards, terminals = rollout[i]
            terminals = self.network.FloatTensor(terminals).unsqueeze(1)
            rewards = self.network.FloatTensor(rewards).unsqueeze(1)
            actions = self.network.LongTensor(actions).unsqueeze(1)
            next_value = rollout[i + 1][2]
            returns = rewards + terminals * config.discount * returns
            td_error = rewards + config.discount * terminals * next_value.data - value.data
            advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [prob, log_prob, value, actions, returns, advantages]

        prob, log_prob, value, actions, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob.gather(1, Variable(actions)) * Variable(advantages)
        policy_loss += config.entropy_weight * torch.sum(prob * log_prob, dim=1, keepdim=True)
        value_loss = config.value_loss_weight * 0.5 * (Variable(returns) - value).pow(2)

        self.optimizer.zero_grad()
        (policy_loss + value_loss).mean().backward()
        nn.utils.clip_grad_norm(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
