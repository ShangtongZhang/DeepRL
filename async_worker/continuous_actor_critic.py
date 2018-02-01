#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import *

class ContinuousAdvantageActorCritic:
    def __init__(self, config, learning_network, extra):
        self.config = config
        self.actor_opt = config.actor_optimizer_fn(learning_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(learning_network.critic.parameters())
        self.worker_network = config.network_fn()
        self.worker_network.load_state_dict(learning_network.state_dict())
        self.task = config.task_fn()
        self.policy = config.policy_fn()
        self.learning_network = learning_network
        self.counter = 0

        self.shared_state_normalizer = extra[0]
        self.state_normalizer = StaticNormalizer(self.task.state_dim)
        self.shared_reward_normalizer = extra[1]
        self.reward_normalizer = StaticNormalizer(1)

    def episode(self, deterministic=False):
        config = self.config
        self.state_normalizer.offline_stats.load(self.shared_state_normalizer.offline_stats)
        self.reward_normalizer.offline_stats.load(self.shared_reward_normalizer.offline_stats)
        state = self.task.reset()
        state = self.state_normalizer(state)
        steps = 0
        total_reward = 0
        pending = []
        while not config.stop_signal.value:
            mean, std, log_std = self.worker_network.actor.predict(np.stack([state]))
            value = self.worker_network.critic.predict(np.stack([state]))
            action = self.policy.sample(mean.data.numpy().flatten(),
                                        std.data.numpy().flatten(),
                                        False)
            action = self.config.action_shift_fn(action)
            next_state, reward, terminal, _ = self.task.step(action)
            next_state = self.state_normalizer(next_state)

            steps += 1
            total_reward += reward
            reward = self.reward_normalizer(reward)

            if deterministic:
                if terminal:
                    break
                state = next_state
                continue

            pending.append([mean, std, log_std, value, action, reward])
            with config.steps_lock:
                config.total_steps.value += 1

            if terminal or len(pending) >= config.update_interval:
                critic_loss = 0
                actor_loss = 0
                if terminal:
                    R = torch.FloatTensor([[0]])
                else:
                    R = self.worker_network.critic(np.stack([next_state])).data
                GAE = torch.FloatTensor([[0]])
                for i in reversed(range(len(pending))):
                    mean, std, log_std, value, action, reward = pending[i]
                    if i == len(pending) - 1:
                        delta = reward + config.discount * R - value.data
                    else:
                        delta = reward + pending[i + 1][3].data - value.data
                    GAE = config.discount * config.gae_tau * GAE + delta

                    action = Variable(torch.FloatTensor([action]))
                    log_density = self.worker_network.actor.log_density(action, mean, log_std, std)
                    actor_loss += -torch.sum(log_density) * Variable(GAE)
                    if config.entropy_weight:
                        actor_loss += -config.entropy_weight * self.worker_network.actor.entropy(std)

                    R = reward + config.discount * R
                    critic_loss += 0.5 * (Variable(R) - value).pow(2)

                pending = []
                self.worker_network.zero_grad()
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm(self.worker_network.parameters(), config.gradient_clip)
                sync_grad(self.learning_network, self.worker_network)
                self.actor_opt.step()
                self.critic_opt.step()
                self.worker_network.load_state_dict(self.learning_network.state_dict())

            if terminal:
                break
            state = next_state

        self.shared_state_normalizer.offline_stats.merge(self.state_normalizer.online_stats)
        self.state_normalizer.online_stats.zero()

        self.shared_reward_normalizer.offline_stats.merge(self.reward_normalizer.online_stats)
        self.reward_normalizer.online_stats.zero()

        return steps, total_reward