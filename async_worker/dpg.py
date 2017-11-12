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
from async_worker import *
import pickle
import os
import time

class DeterministicPolicyGradient:
    def __init__(self, config, shared_network, extra):
        self.config = config
        self.task = config.task_fn()

        self.shared_network = shared_network
        self.worker_network = config.network_fn()
        self.worker_network.load_state_dict(self.shared_network.state_dict())
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.worker_network.state_dict())
        self.actor_opt = config.actor_optimizer_fn(self.shared_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.shared_network.critic.parameters())

        self.random_process = config.random_process_fn()
        self.criterion = nn.MSELoss()

        # self.state_normalizer = Normalizer(self.task.state_dim)
        self.shared_state_normalizer = extra[0]
        self.state_normalizer = StaticNormalizer(self.task.state_dim)
        # self.replay = config.replay_fn()
        self.replay = extra[-1]

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.target_network_mix) +
                                    param.data * self.config.target_network_mix)

    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.state_normalizer(state)

        config = self.config
        actor = self.worker_network.actor
        critic = self.worker_network.critic
        target_actor = self.target_network.actor
        target_critic = self.target_network.critic

        steps = 0
        total_reward = 0.0
        while True:
            actor.eval()
            action = actor.predict(np.stack([state])).flatten()
            if not deterministic:
                action += self.random_process.sample()
            next_state, reward, done, info = self.task.step(action)
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))
            next_state = self.state_normalizer(next_state)
            total_reward += reward

            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                with config.steps_lock:
                    config.total_steps.value += 1

            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.replay.size() >= config.min_memory_size:
                self.worker_network.train()
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                q_next = target_critic.predict(next_states, target_actor.predict(next_states))
                terminals = critic.to_torch_variable(terminals).unsqueeze(1)
                rewards = critic.to_torch_variable(rewards).unsqueeze(1)
                q_next = config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()
                q = critic.predict(states, actions)
                critic_loss = self.criterion(q, q_next)

                critic.zero_grad()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                with config.network_lock:
                    for param, worker_param in zip(self.shared_network.critic.parameters(), critic.parameters()):
                        if param.grad is not None:
                            break
                        param._grad = worker_param.grad
                    self.critic_opt.step()

                actions = actor.predict(states, False)
                var_actions = Variable(actions.data, requires_grad=True)
                q = critic.predict(states, var_actions)
                q.backward(torch.ones(q.size()))

                actor.zero_grad()
                self.actor_opt.zero_grad()
                actions.backward(-var_actions.grad.data)
                with config.network_lock:
                    for param, worker_param in zip(self.shared_network.actor.parameters(), actor.parameters()):
                        if param.grad is not None:
                            break
                        param._grad = worker_param.grad
                    self.actor_opt.step()

                self.worker_network.load_state_dict(self.shared_network.state_dict())

                self.soft_update(self.target_network, self.worker_network)

        return steps, total_reward
