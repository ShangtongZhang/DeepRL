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
        self.learning_network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.learning_network.state_dict())
        self.target_network.eval()
        self.actor_opt = config.actor_optimizer_fn(self.learning_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.learning_network.critic.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.criterion = nn.MSELoss()
        self.total_steps = 0
        self.epsilon = 1.0
        self.d_epsilon = 1.0 / config.noise_decay_interval

        self.state_normalizer = Normalizer(self.task.state_dim)
        self.reward_normalizer = Normalizer(1)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.target_network_mix) +
                                    param.data * self.config.target_network_mix)

    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.state_normalizer(state)

        config = self.config
        actor = self.learning_network.actor
        critic = self.learning_network.critic
        target_actor = self.target_network.actor
        target_critic = self.target_network.critic

        steps = 0
        total_reward = 0.0
        while True:
            actor.eval()
            action = actor.predict(np.stack([state])).flatten()
            if not deterministic:
                if self.total_steps < config.exploration_steps:
                    action = self.task.random_action()
                else:
                    action += max(self.epsilon, config.min_epsilon) * self.random_process.sample()
                    self.epsilon -= self.d_epsilon
            next_state, reward, done, info = self.task.step(action)
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))
            next_state = self.state_normalizer(next_state)
            total_reward += reward
            reward = np.asscalar(self.reward_normalizer(np.array([reward])))

            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.total_steps > config.exploration_steps:
                self.learning_network.train()
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
                critic_loss.backward()
                self.critic_opt.step()

                actor_loss = -critic.predict(states, actor.predict(states, False))
                actor_loss = actor_loss.mean()

                actor.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                self.soft_update(self.target_network, self.learning_network)

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
