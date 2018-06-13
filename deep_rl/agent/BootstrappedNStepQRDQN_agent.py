#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *

class BootstrappedNStepQRDQNAgent(BaseAgent):
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

        self.options = self.network.tensor(np.random.randint(
            config.num_options, size=config.num_workers)).long()
        self.candidate_quantiles = self.network.tensor(config.candidate_quantiles).long()

        self.is_initial_states = np.ones(config.num_workers, dtype=np.uint8)

    def option_to_q_values(self, options, quantiles):
        config = self.config
        if config.smoothed_quantiles:
            if config.num_quantiles % config.num_options:
                raise Exception('Smoothed quantile options is not supported')
            quantiles = quantiles.view(quantiles.size(0), quantiles.size(1), config.num_options, -1)
            quantiles = quantiles.mean(-1)
            q_values = quantiles[self.network.range(quantiles.size(0)), :, options]
        else:
            selected_quantiles = self.candidate_quantiles[options]
            q_values = quantiles[self.network.range(quantiles.size(0)), :, selected_quantiles]
        return q_values

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        quantile_values, option_values = self.network.predict(self.config.state_normalizer(state))
        greedy_options = torch.argmax(option_values, dim=-1)
        if self.config.option_type == 'constant_beta':
            dice = np.random.rand()
            start_new_option = self.info['initial_state'] or dice < self.config.target_beta
            if start_new_option:
                self.info['prev_option'] = greedy_options
            q_values = self.option_to_q_values(self.info['prev_option'], quantile_values)
        elif self.config.option_type is None:
            q_values = quantile_values.sum(-1)
        else:
            raise Exception('Unknown option type')

        q_values = q_values.cpu().detach().numpy()
        self.config.state_normalizer.unset_read_only()
        return np.argmax(q_values.flatten())

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states

        for _ in range(config.rollout_length):
            self.evaluate(config.num_workers)
            self.evaluation_episodes()

            quantile_values, option_values = self.network.predict(self.config.state_normalizer(states))

            greedy_options = torch.argmax(option_values, dim=-1)
            random_option_prob = config.random_option_prob(config.num_workers)
            random_options = self.network.tensor(np.random.randint(
                config.num_options, size=config.num_workers)).long()
            dice = self.network.tensor(np.random.rand(config.num_workers))
            new_options = torch.where(dice < random_option_prob, random_options, greedy_options)

            dice = np.random.rand(config.num_workers)
            start_new_options = np.logical_or(self.is_initial_states, dice < config.behavior_beta)
            start_new_options = self.network.tensor(start_new_options.astype(np.uint8)).byte()

            if config.option_type == 'constant_beta':
                self.options = torch.where(start_new_options, new_options, self.options)
                q_values = self.option_to_q_values(self.options, quantile_values)
            elif config.option_type is None:
                q_values = (quantile_values * self.quantile_weight).sum(-1)
            else:
                raise Exception('Unknown option type')

            q_values = q_values.cpu().detach().numpy()

            actions = [self.policy.sample(v) for v in q_values]
            next_states, rewards, terminals, _ = self.task.step(actions)
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            self.is_initial_states = terminals
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([quantile_values, actions, rewards, 1 - terminals, self.options.clone(), option_values])
            states = next_states

            self.policy.update_epsilon(config.num_workers)
            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        processed_rollout = [None] * (len(rollout))
        quantile_values_next, option_values_next = self.target_network.predict(config.state_normalizer(states))
        a_next = torch.argmax(quantile_values_next.sum(-1), dim=1)
        returns = quantile_values_next[self.network.range(config.num_workers), a_next, :].detach()

        option_values_next = option_values_next.detach()
        option_returns = config.target_beta * torch.max(option_values_next, dim=1)[0] + \
                         (1 - config.target_beta) * option_values_next[self.network.range(config.num_workers), self.options]
        option_returns = option_returns.unsqueeze(1)
        for i in reversed(range(len(rollout))):
            quantile_values, actions, rewards, terminals, options, option_values = rollout[i]
            actions = self.network.tensor(actions).long()
            quantile_values = quantile_values[self.network.tensor(np.arange(config.num_workers)).long(),
                              actions, :]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns
            option_returns = rewards + config.discount * terminals * option_returns
            option_values = option_values[self.network.range(config.num_workers), options].unsqueeze(1)
            processed_rollout[i] = [quantile_values, returns, option_values, option_returns]

        quantile_values, target_quantile_values, option_values, option_returns = map(
            lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        target_quantile_values = target_quantile_values.t().unsqueeze(-1)
        diff = target_quantile_values - quantile_values
        loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()
        loss = loss.mean(1).sum()

        if config.option_type is not None:
            option_loss = (option_values - option_returns).pow(2).mul(0.5).mean()
            loss = loss + option_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

