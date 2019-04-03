#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision

class ResidualDDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.episode_reward = 0
        self.episode_rewards = []

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)
        if self.total_steps < config.min_memory_size:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, _ = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.episode_reward += reward[0]
        reward = self.config.reward_normalizer(reward)
        self.replay.feed([self.state, action, reward, next_state, done.astype(np.uint8)])
        if done[0]:
            config.logger.add_scalar('train_episode_return', self.episode_reward)
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.min_memory_size:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states).squeeze(1)
            actions = tensor(actions).squeeze(1)
            rewards = tensor(rewards)
            next_states = tensor(next_states).squeeze(1)
            terminals = tensor(terminals)
            terminals = 1 - terminals

            if config.target_net_residual:
                target_net = self.target_network
            else:
                target_net = self.network

            if config.symmetric:
                with torch.no_grad():
                    a_next = target_net.actor(next_states)
                    q_next = target_net.critic(next_states, a_next)
                    target = rewards + terminals * config.discount * q_next
                q = self.network.critic(states, actions.detach())
                d_loss = (q - target).pow(2).mul(0.5).mean()

                a_next = self.network.actor(next_states).detach()
                q_next = self.network.critic(next_states, a_next)
                target = rewards + config.discount * terminals * q_next
                with torch.no_grad():
                    q = target_net.critic(states, actions)
                rd_loss = (q - target).pow(2).mul(0.5).mean()

                critic_loss = config.residual * rd_loss + d_loss
            else:
                q = self.network.critic(states, actions)
                with torch.no_grad():
                    a_next = target_net.actor(next_states)
                    q_next = target_net.critic(next_states, a_next)
                    target = rewards + terminals * config.discount * q_next
                    td_error = target - q
                a_next = self.network.actor(next_states).detach()
                q_next = self.network.critic(next_states, a_next)
                critic_loss = (config.residual * config.discount * q_next - q) * td_error
                critic_loss = critic_loss.mean()

            config.logger.add_scalar('q_loss', critic_loss)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()
            config.logger.add_scalar('pi_loss', policy_loss)

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)
