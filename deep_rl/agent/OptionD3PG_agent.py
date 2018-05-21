#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .BaseAgent import *
from ..utils import *

class OptionD3PGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.policy = config.policy_fn()

        self.total_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.random_process = config.random_process_fn((config.num_workers, self.task.action_dim))

        states = self.task.reset()
        states = self.config.state_normalizer(states)
        self.q_options, self.betas, self.actions = self.network.predict(states)
        self.options = np.asarray([self.policy.sample(q) for q in to_numpy(self.q_options)])
        self.is_initial_betas = np.ones(self.config.num_workers)
        self.prev_options = np.copy(self.options)
        self.states = states

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def iteration(self):
        config = self.config
        rollout = []

        states = self.states
        q_options, betas, options, actions = self.q_options, self.betas, self.options, self.actions
        for _ in range(config.rollout_length):
            var_options = self.network.tensor(options).long()
            executed_actions = actions[self.network.range(config.num_workers), var_options]
            executed_actions = to_numpy(executed_actions)
            executed_actions += self.random_process.sample()
            next_states, rewards, terminals, _ = self.task.step(executed_actions)
            next_states = self.config.state_normalizer(next_states)
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            rollout.append([states, executed_actions, options, self.prev_options, rewards, 1 - terminals, np.copy(self.is_initial_betas)])

            q_options_next, betas_next, actions_next = self.network.predict(states)
            self.is_initial_betas = np.asarray(terminals, dtype=np.float32)

            np_q_options_next = to_numpy(q_options_next)
            np_betas_next = betas_next.gather(1, var_options.unsqueeze(1))
            np_betas_next = to_numpy(np_betas_next).flatten()
            options_next = np.copy(options)
            dice = np.random.rand(len(options_next))
            for j in range(len(dice)):
                if dice[j] < np_betas_next[j] or terminals[j]:
                    options_next[j] = self.policy.sample(np_q_options_next[j])

            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            self.prev_options = options
            options = options_next
            q_options = q_options_next
            betas = betas_next
            actions = actions_next
            states = next_states

            self.policy.update_epsilon()
            self.total_steps += config.num_workers

        self.options = options
        self.q_options = q_options
        self.betas = betas
        self.actions = actions
        self.states = states

        target_q_options, _, _ = self.target_network.predict(next_states)

        prev_options = self.network.tensor(self.prev_options).long().unsqueeze(1)
        betas_prev_options = betas.gather(1, prev_options)

        returns = (1 - betas_prev_options) * target_q_options.gather(1, prev_options) +\
                  betas_prev_options * torch.max(target_q_options, dim=1, keepdim=True)[0]
        returns = returns.detach()

        for i in reversed(range(len(rollout))):
            states, actions, options, prev_options, rewards, terminals, is_initial_betas = rollout[i]

            is_initial_betas = self.network.tensor(is_initial_betas)
            prev_options = self.network.tensor(prev_options).unsqueeze(1).long()
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns

            phi = self.network.feature(states)
            actions = self.network.tensor(actions)
            q_options = self.network.critic(phi, actions)
            betas = self.network.termination(phi)
            if not config.off_policy_critic:
                q = q_options[self.network.tensor(np.arange(q_options.size(0))).long(),
                      self.network.tensor(options).long()].unsqueeze(-1)
            else:
                q = q_options
            q_loss = (q - returns).pow(2).mul(0.5).sum(1).mean()

            q_prev_omg = q_options.gather(1, prev_options)
            v_prev_omg = torch.max(q_options, dim=1, keepdim=True)[0]
            advantage_omg = q_prev_omg - v_prev_omg
            advantage_omg.add_(config.termination_regularizer).detach()
            betas = betas.gather(1, prev_options)
            betas = betas * (1 - is_initial_betas)
            beta_loss = (betas * advantage_omg).mean()

            self.network.zero_grad()
            (q_loss + beta_loss * config.beta_loss_weight).backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            actions = self.network.actor(phi)
            q = self.network.critic(phi.detach(), actions)
            if not config.off_policy_actor:
                q = q[self.network.tensor(np.arange(q.size(0))).long(),
                      self.network.tensor(options).long()].unsqueeze(-1)
            policy_loss = -q.sum(1).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)
