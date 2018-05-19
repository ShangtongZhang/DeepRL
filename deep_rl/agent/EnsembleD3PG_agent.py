#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .BaseAgent import *

class EnsembleD3PGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()
        self.states = self.config.state_normalizer(self.states)
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.random_process = config.random_process_fn((config.num_workers, self.task.action_dim))

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            if not config.random_option:
                actions, options = self.network.actor(states, True)
                actions += self.random_process.sample()
            else:
                actions = self.network.actor(states)[0].detach().cpu().numpy()
                options = np.random.randint(0, config.num_options, size=config.num_workers)
                actions = actions[np.arange(config.num_workers), options, :]
            next_states, rewards, terminals, _ = self.task.step(actions)
            next_states = self.config.state_normalizer(next_states)
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([states, actions, rewards, 1 - terminals, next_states, options])
            states = next_states
            self.total_steps += config.num_workers

        self.states = states

        _, q_next, _ = self.target_network.actor(states)
        # returns = q_next.detach()
        returns = q_next.max(1)[0].unsqueeze(1).detach()
        for i in reversed(range(len(rollout))):
            states, actions, rewards, terminals, next_states, options = rollout[i]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns

            q = self.network.critic(states, actions)
            if not config.off_policy_critic:
                q = q[self.network.tensor(np.arange(q.size(0))).long(),
                      self.network.tensor(options).long()].unsqueeze(-1)
                # masked_returns = returns[self.network.tensor(np.arange(returns.size(0))).long(),
                #                          self.network.tensor(options).long()].unsqueeze(-1)
            # else:
            #     masked_returns = returns
            q_loss = (q - returns).pow(2).mul(0.5).sum(1).mean() * config.value_loss_weight

            self.optimizer.zero_grad()
            q_loss.backward()
            self.optimizer.step()

            _, q_values, _ = self.network.actor(states)
            if not config.off_policy_actor:
                q_values = q_values[self.network.tensor(np.arange(q_values.size(0))).long(),
                                    self.network.tensor(options).long()].unsqueeze(-1)
            policy_loss = -q_values.sum(1).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.network.zero_critic_grad()
            self.optimizer.step()
            self.soft_update(self.target_network, self.network)

        self.evaluate(config.rollout_length)