#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
import math

class QuantileEnsembleDDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)
        self.total_steps = 0
        self.cumulative_density = self.network.tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        action = self.network.predict(state, to_numpy=True).flatten()
        self.config.state_normalizer.unset_read_only()
        return action

    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.config.state_normalizer(state)

        config = self.config

        steps = 0
        total_reward = 0.0
        while True:
            self.evaluate()
            self.evaluation_episodes()

            q_values, actions = self.network.predict(np.stack([state]))

            ucb_eps = config.ucb_constant * math.sqrt(math.log(self.total_steps + 1) / (self.total_steps + 1))
            upper_bound = q_values[:, :, -self.config.num_quantiles // 2:].mean(-1)
            q_values = q_values.mean(-1) + ucb_eps * upper_bound

            best = torch.argmax(q_values, dim=-1)
            action = actions[self.network.range(actions.size(0)), best, :]
            action = action.cpu().detach().numpy().flatten()

            action += self.random_process.sample()

            next_state, reward, done, info = self.task.step(action)
            next_state = self.config.state_normalizer(next_state)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)

            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1

            steps += 1
            state = next_state

            if not deterministic and self.replay.size() >= config.min_memory_size:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences

                phi_next = self.target_network.feature(next_states)
                a_next = self.target_network.actor(phi_next)
                q_next = self.target_network.critic(
                    phi_next.unsqueeze(1).expand(-1, a_next.size(1), -1), a_next)

                best_a_next = torch.argmax(q_next.mean(-1), dim=-1)
                q_next = q_next[self.network.range(q_next.size(0)), best_a_next, :]

                terminals = self.network.tensor(terminals).unsqueeze(1)
                rewards = self.network.tensor(rewards).unsqueeze(1)
                q_next = config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()
                phi = self.network.feature(states)
                q = self.network.critic(phi, self.network.tensor(actions))

                diff = q_next.t().unsqueeze(-1) - q
                critic_loss = huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()
                critic_loss = critic_loss.mean(1).mean(0).sum()

                self.network.zero_grad()
                critic_loss.backward()
                self.network.critic_opt.step()

                phi = self.network.feature(states)
                action = self.network.actor(phi)
                policy_loss = -self.network.critic(
                    phi.detach().unsqueeze(1).expand(-1, action.size(1), -1), action).mean(-1).sum(-1).mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)

            if done:
                break

        return total_reward, steps
