#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *
from replay import *
import pickle

class DDPGAgent:
    def __init__(self,
                 task_fn,
                 actor_network_fn,
                 critic_network_fn,
                 actor_optimizer_fn,
                 critic_optimizer_fn,
                 replay_fn,
                 discount,
                 step_limit,
                 tau,
                 exploration_steps,
                 random_process_fn,
                 test_interval,
                 test_repetitions,
                 tag,
                 logger):
        self.task = task_fn()
        self.actor = actor_network_fn()
        self.critic = critic_network_fn()
        self.target_actor = actor_network_fn()
        self.target_critic = critic_network_fn()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = actor_optimizer_fn(self.actor.parameters())
        self.critic_opt = critic_optimizer_fn(self.critic.parameters())
        self.replay = replay_fn()
        self.step_limit = step_limit
        self.tau = tau
        self.logger = logger
        self.discount = discount
        self.exploration_steps = exploration_steps
        self.random_process = random_process_fn()
        self.criterion = nn.MSELoss()
        self.test_interval = test_interval
        self.test_repetitions = test_repetitions
        self.total_steps = 0
        self.tag = tag

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()

        steps = 0
        total_reward = 0.0
        while not self.step_limit or steps < self.step_limit:
            action = self.actor.predict(np.stack([state])).flatten()
            if not deterministic:
                if self.total_steps < self.exploration_steps:
                    action = np.random.uniform(-1, 1, action.shape)
                else:
                    action += self.random_process.sample()
            next_state, reward, done, info = self.task.step(action)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1
            total_reward += reward
            state = next_state

            if done:
                break

            if not deterministic and self.total_steps > self.exploration_steps:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                q_next = self.target_critic.predict(next_states, self.target_actor.predict(next_states))
                terminals = self.critic.to_torch_variable(terminals).unsqueeze(1)
                rewards = self.critic.to_torch_variable(rewards).unsqueeze(1)
                q_next = self.discount * q_next * (1 - terminals)
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
            self.logger.info('episode %d, reward %f, avg reward %f, total steps %d' % (
                ep, reward, avg_reward, self.total_steps))

            if self.test_interval and ep % self.test_interval == 0:
                self.logger.info('Testing...')
                self.save('data/%sddpg-model-%s.bin' % (self.tag, self.task.name))
                test_rewards = []
                for _ in range(self.test_repetitions):
                    test_rewards.append(self.episode(True))
                avg_reward = np.mean(test_rewards)
                avg_test_rewards.append(avg_reward)
                self.logger.info('Avg reward %f(%f)' % (
                    avg_reward, np.std(test_rewards) / np.sqrt(self.test_repetitions)))
                with open('data/%sddpg-statistics-%s.bin' % (self.tag, self.task.name), 'wb') as f:
                    pickle.dump({'rewards': rewards,
                                 'test_rewards': avg_test_rewards}, f)
                if avg_reward > self.task.success_threshold:
                    break
