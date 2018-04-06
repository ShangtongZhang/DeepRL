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

class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self)
        self.config = config
        self.task = config.task_fn()
        self.network = DisjointActorCriticNet(self.task.state_dim, self.task.action_dim,
                                              config.actor_network_fn, config.critic_network_fn)
        self.actor = self.network.actor
        self.critic = self.network.critic
        self.actor_opt = config.actor_optimizer_fn(self.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.critic.parameters())
        self.total_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for i in range(config.rollout_length):
            mean, std, log_std = self.actor.predict(states)
            values = self.critic.predict(states)
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).detach()
            log_probs = torch.sum(log_probs, dim=1, keepdim=True)
            next_states, rewards, terminals, _ = self.task.step(actions.data.cpu().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values, actions, log_probs, rewards, 1 - terminals])
            states = next_states

        self.states = states
        pending_value = self.critic.predict(states)
        rollout.append([states, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = self.actor.tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.data
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = self.actor.tensor(terminals).unsqueeze(1)
            rewards = self.actor.tensor(rewards).unsqueeze(1)
            actions = self.actor.variable(actions)
            states = self.actor.variable(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.data
            else:
                td_error = rewards + config.discount * terminals * next_value.data - value.data
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()
        advantages = Variable(advantages)

        for k in range(config.optimization_epochs):
            mean, std, log_std = self.actor.predict(states)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions)
            log_probs = torch.sum(log_probs, dim=1, keepdim=True)
            ratio = (log_probs - log_probs_old).exp()
            obj = ratio * advantages
            obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip, 1.0 + self.config.ppo_ratio_clip) * advantages
            policy_loss = -torch.min(obj, obj_clipped).mean(0)

            v = self.critic.predict(states)
            value_loss = 0.5 * (Variable(returns) - v).pow(2).mean()

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            nn.utils.clip_grad_norm(self.actor.parameters(), config.gradient_clip)
            nn.utils.clip_grad_norm(self.critic.parameters(), config.gradient_clip)
            self.actor_opt.step()
            self.critic_opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
