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

                actions = actor.predict(states, False)
                var_actions = Variable(actions.data, requires_grad=True)
                q = critic.predict(states, var_actions)
                q.backward(torch.ones(q.size()))

                actor.zero_grad()
                actions.backward(-var_actions.grad.data)
                self.actor_opt.step()

                self.soft_update(self.target_network, self.learning_network)

        return total_reward, steps

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.learning_network.state_dict(), f)
