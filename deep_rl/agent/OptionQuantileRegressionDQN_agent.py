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

class OptionQuantileRegressionDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0

        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = self.network.tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))
        candidate_quantile = np.linspace(0.1, 0.9, config.num_options) * config.num_quantiles
        if config.mean_option:
            candidate_quantile = candidate_quantile.tolist() + [config.num_quantiles]
            candidate_quantile = np.asarray([candidate_quantile])
        self.candidate_quantile = self.network.tensor(candidate_quantile).long().expand(
            config.num_workers, -1)

    def huber(self, x):
        cond = (x < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def act(self, quantile_values, pi):
        dist = torch.distributions.Categorical(pi)
        option = dist.sample()
        option_quantiles = self.candidate_quantile[self.network.range(option.size(0)), option]
        if self.config.mean_option:
            mean_q_values = quantile_values.mean(-1).unsqueeze(-1)
            quantile_values = torch.cat([quantile_values, mean_q_values], dim=-1)
        q_values = quantile_values[self.network.range(option.size(0)), :, option_quantiles]
        q_values = q_values.cpu().detach().numpy()
        actions = [self.policy.sample(v) for v in q_values]
        return actions, option

    def evaluation_action(self, state):
        value = self.network.predict(np.stack([self.config.state_normalizer(state)])).squeeze(0).detach()
        value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
        return np.argmax(value)

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while True:
            quantile_value, pi, v_pi = self.network.predict(np.stack([self.config.state_normalizer(state)]))
            action, option = self.act(quantile_value, pi)
            action = action[0]
            if self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, self.task.action_dim)
            next_state, reward, done, _ = self.task.step(action)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1

            if not deterministic and self.total_steps > self.config.exploration_steps:
                _, _, v_pi_next = self.target_network.predict(np.stack([self.config.state_normalizer(next_state)]))
                v_pi_next = v_pi_next.detach()
                v_pi_next.mul_(self.config.discount * (1 - done)).add_(reward)
                td_error = v_pi_next - v_pi
                log_pi = pi.log()
                option_pi_loss = -log_pi[0, option.unsqueeze(-1)].unsqueeze(-1) * td_error.detach()
                option_pi_loss = option_pi_loss.mean()
                option_entropy_loss = (pi * log_pi).sum(-1).mean()
                option_v_loss = 0.5 * td_error.pow(2).mean()
                self.optimizer.zero_grad()
                (option_pi_loss + option_v_loss + self.config.entropy_weight * option_entropy_loss).backward()
                self.optimizer.step()

                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)

                quantiles_next = self.target_network.predict(next_states)[0].detach()
                q_next = (quantiles_next * self.quantile_weight).sum(-1)
                _, a_next = torch.max(q_next, dim=1)
                a_next = a_next.view(-1, 1, 1).expand(-1, -1, quantiles_next.size(2))
                quantiles_next = quantiles_next.gather(1, a_next).squeeze(1)

                rewards = self.network.tensor(rewards)
                terminals = self.network.tensor(terminals)
                quantiles_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * quantiles_next

                quantiles = self.network.predict(states)[0]
                actions = self.network.tensor(actions).long()
                actions = actions.view(-1, 1, 1).expand(-1, -1, quantiles.size(2))
                quantiles = quantiles.gather(1, actions).squeeze(1)

                quantiles_next = quantiles_next.t().unsqueeze(-1)
                diff = quantiles_next - quantiles
                loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()

                self.optimizer.zero_grad()
                loss.mean(1).sum().backward()
                self.optimizer.step()

            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()

            if done:
                break
            state = next_state

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps
