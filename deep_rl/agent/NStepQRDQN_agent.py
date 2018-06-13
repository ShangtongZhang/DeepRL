#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *

class NStepQRDQNAgent(BaseAgent):
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
        self.info = {'option': -1}

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        quantile_values = self.network.predict(self.config.state_normalizer(state))
        q_values = quantile_values.sum(-1).cpu().detach().numpy()
        self.config.state_normalizer.unset_read_only()
        return np.argmax(q_values.flatten())

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states

        for _ in range(config.rollout_length):
            self.evaluate(config.rollout_length)
            self.evaluation_episodes()

            quantile_values = self.network.predict(self.config.state_normalizer(states))
            if self.config.random_option:
                options = self.network.tensor(np.random.randint(
                    0, config.num_quantiles, size=config.num_workers)).long()
                q_values = quantile_values[self.network.range(config.num_workers), :, options].cpu().detach().numpy()
            else:
                q_values = (quantile_values * self.quantile_weight).sum(-1).cpu().detach().numpy()
            actions = [self.policy.sample(v) for v in q_values]
            next_states, rewards, terminals, _ = self.task.step(actions)
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([quantile_values, actions, rewards, 1 - terminals])
            states = next_states

            self.policy.update_epsilon()
            self.total_steps += config.num_workers
            if self.total_steps / config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        processed_rollout = [None] * (len(rollout))
        quantile_values_next = self.target_network.predict(config.state_normalizer(states))
        a_next = torch.max(quantile_values_next.sum(-1), dim=1)[1]
        returns = quantile_values_next[self.network.tensor(np.arange(config.num_workers)).long(),
                  a_next, :].detach()
        for i in reversed(range(len(rollout))):
            quantile_values, actions, rewards, terminals = rollout[i]
            actions = self.network.tensor(actions).long()
            quantile_values = quantile_values[self.network.tensor(np.arange(config.num_workers)).long(),
                              actions, :]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns
            processed_rollout[i] = [quantile_values, returns]

        quantile_values, target_quantile_values = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        target_quantile_values = target_quantile_values.t().unsqueeze(-1)
        diff = target_quantile_values - quantile_values
        loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()

        self.optimizer.zero_grad()
        loss.mean(1).sum().backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

