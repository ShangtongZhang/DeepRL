#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *
from component import *
from utils import *
import pickle
import torch.nn as nn

class DDPGAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.actor = config.actor_network_fn()
        self.critic = config.critic_network_fn()
        self.target_actor = config.actor_network_fn()
        self.target_critic = config.critic_network_fn()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.eval()
        self.target_critic.eval()
        self.actor_opt = config.actor_optimizer_fn(self.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.critic.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.criterion = nn.MSELoss()
        self.total_steps = 0
        self.epsilon = 1.0
        self.d_epsilon = 1.0 / config.noise_decay_interval

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.target_network_mix) +
                                    param.data * self.config.target_network_mix)

    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.config.state_shift_fn(state)

        steps = 0
        total_reward = 0.0
        while not self.config or steps < self.config.max_episode_length:
            self.actor.eval()
            action = self.actor.predict(np.stack([state])).flatten()
            self.config.logger.histo_summary('state', state, self.total_steps)
            self.config.logger.histo_summary('action', action, self.total_steps)
            self.config.logger.histo_summary('layer1_act', self.actor.layer1_act, self.total_steps)
            self.config.logger.histo_summary('layer2_act', self.actor.layer2_act, self.total_steps)
            self.config.logger.histo_summary('layer3_act', self.actor.layer3_act, self.total_steps)
            self.config.logger.histo_summary('layer1_weight', self.actor.layer1_w, self.total_steps)
            self.config.logger.histo_summary('layer2_weight', self.actor.layer2_w, self.total_steps)
            self.config.logger.histo_summary('layer3_weight', self.actor.layer3_w, self.total_steps)
            if not deterministic:
                if self.total_steps < self.config.exploration_steps:
                    action = self.task.random_action()
                else:
                    action += max(self.epsilon, 0) * self.random_process.sample()
                    self.epsilon -= self.d_epsilon
            self.config.logger.histo_summary('noised action', action, self.total_steps)
            action = self.config.action_shift_fn(action)
            next_state, reward, done, info = self.task.step(action)
            next_state = self.config.state_shift_fn(next_state)
            self.config.logger.scalar_summary('reward', reward, self.total_steps)
            total_reward += reward
            reward = self.config.reward_shift_fn(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.actor.train()
                self.critic.train()
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                q_next = self.target_critic.predict(next_states, self.target_actor.predict(next_states))
                terminals = self.critic.to_torch_variable(terminals).unsqueeze(1)
                rewards = self.critic.to_torch_variable(rewards).unsqueeze(1)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = Variable(q_next.data)
                q = self.critic.predict(states, actions)
                critic_loss = self.criterion(q, q_next)

                self.critic.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                actor_loss = -self.critic.predict(states, self.actor.predict(states, False))
                actor_loss = actor_loss.mean()

                self.actor.zero_grad()
                actor_loss.backward()
                self.config.logger.histo_summary('layer1_g', self.actor.layer1.weight.grad.data.numpy(), self.total_steps)
                self.config.logger.histo_summary('layer2_g', self.actor.layer2.weight.grad.data.numpy(), self.total_steps)
                self.config.logger.histo_summary('layer3_g', self.actor.layer3.weight.grad.data.numpy(), self.total_steps)
                self.actor_opt.step()

                self.soft_update(self.target_actor, self.actor)
                self.soft_update(self.target_critic, self.critic)

        return total_reward

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.actor.state_dict(), f)

    def run(self):
        window_size = 100
        ep = 0
        rewards = []
        avg_test_rewards = []
        while True:
            ep += 1
            reward = self.episode()
            rewards.append(reward)
            avg_reward = np.mean(rewards[-window_size:])
            self.config.logger.info('episode %d, reward %f, avg reward %f, total steps %d' % (
                ep, reward, avg_reward, self.total_steps))

            if self.config.test_interval and ep % self.config.test_interval == 0:
                self.config.logger.info('Testing...')
                with open('data/%s-ddpg-model-%s.bin' % (self.config.tag, self.task.name), 'wb') as f:
                    pickle.dump(self.actor.state_dict(), f)
                test_rewards = []
                for _ in range(self.config.test_repetitions):
                    test_rewards.append(self.episode(True))
                avg_reward = np.mean(test_rewards)
                avg_test_rewards.append(avg_reward)
                self.config.logger.info('Avg reward %f(%f)' % (
                    avg_reward, np.std(test_rewards) / np.sqrt(self.config.test_repetitions)))
                with open('data/%s-ddpg-statistics-%s.bin' % (self.config.tag, self.task.name), 'wb') as f:
                    pickle.dump({'rewards': rewards,
                                 'test_rewards': avg_test_rewards}, f)
                if avg_reward > self.task.success_threshold:
                    break
