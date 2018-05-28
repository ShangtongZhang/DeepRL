#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *

class OptionNStepQRDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())
        self.policy = config.policy_fn()

        self.total_steps = 0
        self.states = self.task.reset()
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

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

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        quantile_values, pi, v_pi = self.network.predict(self.config.state_normalizer(state))
        actions, _ = self.act(quantile_values, pi, True)
        self.config.state_normalizer.unset_read_only()
        return actions[0]

    def act(self, quantile_values, pi, deterministic=False):
        if self.config.random_option:
            option = self.network.tensor([np.random.randint(0, pi.size(1))]).long()
        elif deterministic:
            option = torch.argmax(pi, dim=-1)
        else:
            dist = torch.distributions.Categorical(pi)
            option = dist.sample()
        option_quantiles = self.candidate_quantile[self.network.range(option.size(0)), option]
        if self.config.mean_option:
            mean_q_values = quantile_values.mean(-1).unsqueeze(-1)
            quantile_values = torch.cat([quantile_values, mean_q_values], dim=-1)
        q_values = quantile_values[self.network.range(option.size(0)), :, option_quantiles]
        q_values = q_values.cpu().detach().numpy()
        actions = [self.policy.sample(v, deterministic=deterministic) for v in q_values]
        return actions, option

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            quantile_values, pi, v_pi = self.network.predict(self.config.state_normalizer(states))
            actions, options = self.act(quantile_values, pi)
            # for i in range(config.num_workers):
            #     config.logger.scalar_summary('worker_%d' % (i), options[i])
            # config.logger.scalar_summary('mean_eipsode_reward', np.mean(self.last_episode_rewards))
            next_states, rewards, terminals, _ = self.task.step(actions)
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([quantile_values, actions, rewards, 1 - terminals, options, pi, v_pi])
            states = next_states

            self.policy.update_epsilon()
            self.total_steps += config.num_workers
            if self.total_steps / config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        processed_rollout = [None] * (len(rollout))
        quantile_values_next, _, v_option = self.target_network.predict(config.state_normalizer(states))
        a_next = torch.max(quantile_values_next.sum(-1), dim=1)[1]
        returns = quantile_values_next[self.network.range(config.num_workers),
                  a_next, :].detach()
        option_returns = v_option.detach()
        for i in reversed(range(len(rollout))):
            quantile_values, actions, rewards, terminals, options, pi, v_pi = rollout[i]
            actions = self.network.tensor(actions).long()
            quantile_values = quantile_values[self.network.tensor(np.arange(config.num_workers)).long(),
                              actions, :]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns
            option_returns = rewards + config.discount * terminals * option_returns
            processed_rollout[i] = [quantile_values, returns, options, pi, v_pi, option_returns]

        quantile_values, target_quantile_values, options, pi, v_pi, option_returns \
            = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        target_quantile_values = target_quantile_values.t().unsqueeze(-1)
        diff = target_quantile_values - quantile_values
        loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()
        loss = loss.mean(1).sum()
        td_error = option_returns - v_pi
        log_pi = pi.log()
        option_pi_loss = -log_pi[self.network.range(options.size(0)), options].unsqueeze(-1) * td_error.detach()
        option_pi_loss = option_pi_loss.mean()
        option_v_loss = 0.5 * td_error.pow(2)
        option_v_loss = option_v_loss.mean()
        option_entropy_loss = (pi * log_pi).sum(-1).mean()

        self.optimizer.zero_grad()
        (loss + option_pi_loss + option_v_loss + config.entropy_weight * option_entropy_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        self.evaluate(config.rollout_length)