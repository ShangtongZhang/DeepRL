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

class PPOWorker:
    def __init__(self, config, shared_network, shared_state_shifter):
        self.config = config
        # self.shared_network = shared_network
        # self.local_netwrok = config.network_fn()
        self.task = config.task_fn()
        self.policy = config.policy_fn()

        self.actor_net = config.actor_network_fn()
        self.critic_net = config.critic_network_fn()

        self.actor_opt = config.actor_optimizer_fn(self.actor_net.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.critic_net.parameters())

        # self.shared_state_shifter()

    def rollout(self, deterministic=False):
        config = self.config
        replay = config.replay_fn()
        state = self.task.reset()
        episode_length = 0
        episode_reward = 0
        reward_history = []

        actor_net_old = config.actor_network_fn()
        actor_net_old.load_state_dict(self.actor_net.state_dict())
        ep_count = 0

        while True:
            # self.local_netwrok.load_state_dict(self.shared_network.state_dict())

            batched_episode = 0
            while not replay.full():
                states = []
                actions = []
                rewards = []
                values = []
                returns = []
                advantages = []

                for i in range(config.rollout_length):
                    mean, std, log_std = self.actor_net.predict(np.stack([state]))
                    value = self.critic_net.predict(np.stack([state]))
                    action = self.policy.sample(mean.data.numpy().flatten(), std.data.numpy().flatten(), deterministic)
                    action = self.config.action_shift_fn(action)
                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    state, reward, done, _ = self.task.step(action)
                    episode_reward += reward
                    episode_length += 1
                    done = (done or (config.max_episode_length and episode_length > config.max_episode_length))
                    reward = self.config.reward_shift_fn(reward)
                    rewards.append(reward)

                    if done:
                        episode_length = 0
                        batched_episode += 1
                        reward_history.append(episode_reward)
                        # print episode_reward, np.mean(reward_history[-100:])
                        # episode_reward = 0
                        state = self.task.reset()
                        break

                R = torch.zeros((1, 1))
                if not done:
                    R = self.critic_net.predict(np.stack([state])).data

                values.append(Variable(R))
                A = Variable(torch.zeros((1, 1)))
                for i in reversed(range(len(rewards))):
                    R = Variable(torch.FloatTensor([[rewards[i]]]))
                    ret = R + self.config.discount * values[i + 1]
                    A = ret - values[i] + self.config.discount * self.config.gae_tau * A
                    advantages.append(A.detach())
                    returns.append(ret.detach())
                advantages = list(reversed(advantages))
                returns = list(reversed(returns))
                replay.feed([states, actions, returns, advantages])

            episode_reward /= batched_episode
            print ep_count, episode_reward
            ep_count += 1

            for _ in np.arange(self.config.optimize_epochs):
                # local_network.load_state_dict(self.shared_network.state_dict())

                states, actions, returns, advantages = replay.sample()
                states = self.actor_net.to_torch_variable(np.stack(states))
                actions = self.actor_net.to_torch_variable(np.stack(actions))
                returns = torch.cat(returns, 0)
                advantages = torch.cat(advantages, 0)
                advantages = (advantages - advantages.mean().expand_as(advantages)) / advantages.std().expand_as(advantages)

                mean_old, std_old, log_std_old = actor_net_old.predict(states)
                probs_old = self.actor_net.log_density(actions, mean_old, log_std_old, std_old)
                mean, std, log_std = self.actor_net.predict(states)
                probs = self.actor_net.log_density(actions, mean, log_std, std)
                ratio = (probs - probs_old).exp()
                obj = ratio * advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip, 1.0 + self.config.ppo_ratio_clip) * advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0)
                if config.entropy_weight:
                    policy_loss += config.entropy_weight * self.actor_net.kl_loss(std)

                v = self.critic_net.predict(states)
                value_loss = 0.5 * (returns - v).pow(2).mean()

                self.critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm(self.critic_net.parameters(), config.gradient_clip)
                self.critic_opt.step()


                actor_net_old.load_state_dict(self.actor_net.state_dict())
                self.actor_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm(self.actor_net.parameters(), config.gradient_clip)
                self.actor_opt.step()

            replay.clear()

class PPOAgent:
    def __init__(self, config):
        self.config = config

    def run(self):
        worker = PPOWorker(self.config, None, None)
        worker.rollout()
