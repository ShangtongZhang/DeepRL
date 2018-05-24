#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .BaseAgent import *
from ..utils import *

class GammaDDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.opt = config.optimizer_fn(self.network.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)
        self.total_steps = 0
        self.gammas = self.network.tensor(config.gammas).unsqueeze(0)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        actions, q_options = self.network.predict(np.stack([state]))
        option = np.asscalar(to_numpy(torch.argmax(q_options)))
        action = to_numpy(actions[option]).flatten()
        return action

    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.config.state_normalizer(state)

        config = self.config

        steps = 0
        total_reward = 0.0
        while True:
            actions, q_options = self.network.predict(np.stack([state]))
            option = np.asscalar(to_numpy(torch.argmax(q_options, dim=1)))
            action = to_numpy(actions[option])
            if not deterministic:
                action += self.random_process.sample()
            action = action.flatten()
            next_state, reward, done, info = self.task.step(action)
            next_state = self.config.state_normalizer(next_state)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)

            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done), option])
                self.total_steps += 1

            steps += 1
            state = next_state

            self.evaluate()

            if not deterministic and self.replay.size() >= config.min_memory_size:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, options = experiences
                phi_next = self.target_network.feature(next_states)
                a_next, q_options_next = self.target_network.actor(phi_next)
                if config.target_type == 'vanilla':
                    q_next = self.target_network.critic(phi_next, a_next)
                elif config.target_type == 'mixed':
                    actions_next = torch.stack(a_next).transpose(0, 1)
                    phi_next = phi_next.unsqueeze(1).expand(-1, actions_next.size(1), -1)
                    q_next = self.network.critic(phi_next, actions_next)
                    q_next = q_next.max(1)[0]
                # q_next = q_next.max(1)[0].unsqueeze(1)
                terminals = self.network.tensor(terminals).unsqueeze(1)
                rewards = self.network.tensor(rewards).unsqueeze(1)
                q_next = self.gammas * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()

                q_options_next = q_options_next.max(1)[0].unsqueeze(-1).detach()
                q_options_next = config.discount * (1 - terminals) * q_options_next
                q_options_next.add_(rewards)

                phi = self.network.feature(states)
                _, q_options = self.network.actor(phi)
                options = self.network.tensor(options).long()
                q_options = q_options[self.network.range(q_options.size(0)), options]
                q_option_loss = (q_options_next - q_options).pow(2).mul(0.5).mean()

                actions = self.network.tensor(actions)
                q = self.network.critic(phi, actions)
                q_loss = (q - q_next).pow(2).mul(0.5).sum(1).mean()

                self.network.zero_grad()
                (q_option_loss + q_loss.mul(config.critic_loss_weight)).backward()
                # self.network.critic_opt.step()
                self.opt.step()

                phi = self.network.feature(states)
                actions, _ = self.network.actor(phi)
                q = self.network.critic(phi.detach(), actions)
                policy_loss = -q.sum(1).mean()
                self.network.zero_grad()
                policy_loss.backward()
                self.network.zero_non_actor_grad()
                self.opt.step()

                self.soft_update(self.target_network, self.network)

            if done:
                break

        return total_reward, steps
