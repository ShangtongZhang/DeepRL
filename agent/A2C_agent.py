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
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

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

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for i in range(config.rollout_length):
            prob, log_prob, value = self.network.predict(states)
            actions = [self.policy.sample(p) for p in prob.data.cpu().numpy()]
            actions = config.action_shift_fn(actions)
            next_states, rewards, terminals, _ = self.task.step(actions)
            self.episode_rewards += rewards
            rewards = config.reward_shift_fn(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    next_states[i] = self.task.reset(i)
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([prob, log_prob, value, actions, rewards, 1 - terminals])
            states = next_states

        self.states = states
        _, _, pending_value = self.network.predict(states)
        rollout.append([None, None, pending_value, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = self.network.tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.data
        for i in reversed(range(len(rollout) - 1)):
            prob, log_prob, value, actions, rewards, terminals = rollout[i]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            actions = self.network.tensor(actions, torch.LongTensor).unsqueeze(1)
            next_value = rollout[i + 1][2]
            returns = rewards + config.discount * terminals * returns
            if config.use_gae:
                advantages = returns - value.data
            else:
                td_error = rewards + config.discount * terminals * next_value.data - value.data
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [prob, log_prob, value, actions, returns, advantages]

        prob, log_prob, value, actions, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob.gather(1, Variable(actions)) * Variable(advantages)
        entropy_loss = torch.sum(prob * log_prob, dim=1, keepdim=True)
        value_loss = 0.5 * (Variable(returns) - value).pow(2)

        self.policy_loss = np.mean(policy_loss.data.cpu().numpy())
        self.entropy_loss = np.mean(entropy_loss.data.cpu().numpy())
        self.value_loss = np.mean(value_loss.data.cpu().numpy())

        self.optimizer.zero_grad()
        (policy_loss + config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).mean().backward()
        nn.utils.clip_grad_norm(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
