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

class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            q_values = self._network(config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            mask = 1 - tensor(terminals)
            rewards = tensor(rewards)
            actions = tensor(actions).long()

            if config.target_net_residual:
                target_net = self.target_network
            else:
                target_net = self.network

            with torch.no_grad():
                q_next = target_net(next_states).detach().max(1)[0]
                q_next = rewards + config.discount * q_next * mask
            q = self.network(states)
            q = q[self.batch_indices, actions]
            d_loss = (q - q_next).pow(2).mul(0.5).mean()

            q_next = self.network(next_states).max(1)[0]
            q_next = rewards + config.discount * q_next * mask
            with torch.no_grad():
                q = target_net(states)
                q = q[self.batch_indices, actions]
            rd_loss = (q - q_next).pow(2).mul(0.5)

            if config.r_aware:
                residual = tensor(np.ones(rd_loss.size())).mul(config.residual)
                residual = (rewards != 0).float() * residual

            rd_loss = (rd_loss * residual).mean()

            loss = rd_loss + d_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
