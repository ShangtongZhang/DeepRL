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
from .BaseAgent import *

class PlanDDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.opt = config.optimizer_fn(self.network.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)
        self.total_steps = 0

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        action = self.network.predict(state, to_numpy=True).flatten()
        self.config.state_normalizer.unset_read_only()
        return action

    def episode(self, deterministic=False, video_recorder=None):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.config.state_normalizer(state)

        config = self.config

        steps = 0
        total_reward = 0.0
        while True:
            action = self.network.predict(np.stack([state]), True).flatten()
            if not deterministic:
                action += self.random_process.sample()
            next_state, reward, done, info = self.task.step(action)
            if video_recorder is not None:
                video_recorder.capture_frame()
            next_state = self.config.state_normalizer(next_state)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)

            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1

            steps += 1
            state = next_state

            self.evaluate()

            if not deterministic and self.replay.size() >= config.min_memory_size:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                q_next, _ = self.target_network.critic(next_states, self.target_network.actor(next_states))
                terminals = self.network.tensor(terminals).unsqueeze(1)
                rewards = self.network.tensor(rewards).unsqueeze(1)
                q_next = config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()
                q, r = self.network.critic(states, actions)
                q_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
                r_loss = (r - rewards).pow(2).mul(0.5).mean()
                # config.logger.scalar_summary('q_loss', q_loss, self.total_steps)
                # config.logger.scalar_summary('reward_loss', r_loss, self.total_steps)

                self.opt.zero_grad()
                q_loss.backward()
                # (q_loss + r_loss).backward()
                self.opt.step()

                dead_actions = self.network.actor(states)
                dead_actions.detach_().requires_grad_()
                q = self.network.critic(states, dead_actions)[0].mean()

                self.opt.zero_grad()
                q.backward()

                actions = self.network.actor(states)
                self.opt.zero_grad()
                actions.backward(-dead_actions.grad.detach())
                self.opt.step()

                self.soft_update(self.target_network, self.network)

            if done:
                break

        return total_reward, steps
