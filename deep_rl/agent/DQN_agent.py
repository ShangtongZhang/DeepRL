#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.replay = config.replay_fn()
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.state = self.task.reset()

        self.batch_indices = range_tensor(self.replay.batch_size)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        q = self.network(state)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        q_values = self.network(config.state_normalizer(np.stack([self.state])))
        q_values = to_np(q_values).flatten()
        if self.total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, config.action_dim)
        else:
            action = np.argmax(q_values)
        next_state, reward, done, _ = self.task.step(action)
        self.episode_reward += reward
        self.total_steps += 1
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            next_state = self.task.reset()
        reward = config.reward_normalizer(reward)
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        self.state = next_state

        if self.total_steps > self.config.exploration_steps \
                and self.total_steps % self.config.sgd_update_frequency == 0:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network(next_states).detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q = self.network(states)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        if self.total_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
