#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision

class DoubleDDPGAgent(BaseAgent):
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
        action = action[:, -1, :]
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
            action = action[:, -1, :]
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, _ = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.episode_reward += reward[0]
        reward = self.config.reward_normalizer(reward)

        bootstrap_mask = np.random.rand(2) < config.bootstrap_prob
        self.replay.feed([self.state, action, reward, next_state, done.astype(np.uint8), bootstrap_mask.astype(np.uint8)])
        if done[0]:
            config.logger.add_scalar('train_episode_return', self.episode_reward)
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.min_memory_size:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, mask, b_mask = experiences
            states = states.squeeze(1)
            actions = actions.squeeze(1)
            rewards = tensor(rewards)
            next_states = next_states.squeeze(1)
            mask = tensor(mask)
            b_mask = tensor(b_mask)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            a_next = torch.cat([a_next[:, [1], :], a_next[:, [0], :]], dim=1)
            phi_next = phi_next.unsqueeze(1).expand(-1, 2, -1)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = q_next.min(-1)[0]
            q_next = config.discount * q_next * (1 - mask)
            q_next.add_(rewards)
            q_next = q_next.detach()

            phi = self.network.feature(states)
            q = self.network.critic(phi, tensor(actions))
            critic_loss = (q - q_next).mul(b_mask).pow(2).mul(0.5).sum(-1).mean()
            config.logger.add_scalar('q_std', q.std(-1).mean())
            config.logger.add_histogram('q_std_dist', q[0])
            config.logger.add_scalar('q_loss', critic_loss)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)

            policy_loss = 0
            for i in range(2):
                q = self.network.critic(phi.detach(), action[:, i, :])
                q = q[:, i]
                policy_loss = policy_loss - q.mean()
            q = self.network.critic(phi.detach(), action[:, -1, :])
            policy_loss = policy_loss - q.mean()

            config.logger.add_scalar('pi_loss', policy_loss)

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)
