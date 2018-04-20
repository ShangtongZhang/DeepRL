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

class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = DisjointActorCriticWrapper(self.task.state_dim, self.task.action_dim,
                                              config.actor_network_fn, config.critic_network_fn)
        self.actor = self.network.actor
        self.critic = self.network.critic
        self.target_network = DisjointActorCriticWrapper(self.task.state_dim, self.task.action_dim,
                                              config.actor_network_fn, config.critic_network_fn)
        self.target_network.load_state_dict(self.network.state_dict())
        self.actor_opt = config.actor_optimizer_fn(self.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.critic.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)
        self.criterion = nn.MSELoss()
        self.total_steps = 0

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.target_network_mix) +
                                    param.data * self.config.target_network_mix)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        action = self.actor.predict(state, to_numpy=True).flatten()
        self.config.state_normalizer.unset_read_only()
        return action

    def episode(self, deterministic=False, video_recorder=None):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.config.state_normalizer(state)

        config = self.config
        actor = self.network.actor
        critic = self.network.critic
        target_actor = self.target_network.actor
        target_critic = self.target_network.critic

        steps = 0
        total_reward = 0.0
        while True:
            action = actor.predict(np.stack([state]), True).flatten()
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
                q_next = target_critic.predict(next_states, target_actor.predict(next_states))
                terminals = critic.variable(terminals).unsqueeze(1)
                rewards = critic.variable(rewards).unsqueeze(1)
                q_next = config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()
                q = critic.predict(states, actions)
                critic_loss = self.criterion(q, q_next)

                critic.zero_grad()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                actions = actor.predict(states, False)
                var_actions = Variable(actions.data, requires_grad=True)
                q = critic.predict(states, var_actions)
                q.backward(critic.tensor(np.ones(q.size())))

                actor.zero_grad()
                self.actor_opt.zero_grad()
                actions.backward(-var_actions.grad.data)
                for param in actor.parameters():
                    param.grad.data.clamp(-config.gradient_clip, config.gradient_clip)
                self.actor_opt.step()

                self.soft_update(self.target_network, self.network)

            if done:
                break

        return total_reward, steps
