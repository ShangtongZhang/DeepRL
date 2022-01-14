#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class PPOC51Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.task_extra = config.task_fn()
        self.network = config.network_fn()
        if config.shared_repr:
            self.opt = config.optimizer_fn(self.network.parameters())
        else:
            self.actor_opt = config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        if config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda step: 1 - step / config.max_steps)

        atoms = np.linspace(config.categorical_v_min,
                                   config.categorical_v_max, config.categorical_n_atoms)
        terminal_prob = np.zeros(atoms.shape)
        terminal_prob[np.argmin(np.abs(atoms))] = 1
        self.atoms = tensor(atoms)
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)

    def compute_v(self, v_prob):
        return (v_prob * self.atoms.view(1, -1)).sum(-1).unsqueeze(-1)

    def augment_prediction(self, prediction):
        if self.config.aux:
            prediction['v'] = prediction['v_canonical']
        else:
            prediction['v'] = self.compute_v(prediction['v_prob'])

    def collect_extra_data(self):
        config = self.config
        self.config.state_normalizer.set_read_only()
        states = self.task_extra.reset()
        states = self.config.state_normalizer(states)
        storage = Storage(config.extra_data * config.rollout_length)
        for _ in range(config.extra_data * config.rollout_length):
            prediction = self.network(states)
            self.augment_prediction(prediction)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            mask = tensor(1 - terminals).unsqueeze(-1)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': mask,
                         's': tensor(states),
                         'next_s': tensor(next_states),
                         })
            states = next_states
        prediction = self.network(states)
        self.augment_prediction(prediction)
        storage.add(prediction)
        self.config.state_normalizer.unset_read_only()
        return storage

    def train_with_extra_data(self, storage):
        config = self.config
        storage.placeholder()

        states, rewards, next_states, masks = storage.cat(['s', 'r', 'next_s', 'm'])
        for _ in range(config.optimization_epochs):
            sampler = random_sample(
                np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_next_states = next_states[batch_indices]
                sampled_rewards = rewards[batch_indices]
                sampled_masks = masks[batch_indices]

                prediction = self.network(sampled_states)
                with torch.no_grad():
                    next_prediction = self.network(sampled_next_states)
                atoms_target = sampled_rewards + config.discount * sampled_masks * self.atoms.view(1, -1)
                atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
                next_prob = next_prediction['v_prob']
                atoms_target = atoms_target.unsqueeze(1)
                target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * next_prob.unsqueeze(1)
                target_prob = target_prob.sum(-1)

                value_loss = -(target_prob * prediction['v_log_prob']).sum(-1).mean()

                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()

    def step(self):
        config = self.config
        if config.extra_data:
            storage = self.collect_extra_data()
            self.train_with_extra_data(storage)
        storage = Storage(config.rollout_length, ['t_atoms', 't_probs', 'v_canonical'])
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            self.augment_prediction(prediction)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states),
                         'next_s': tensor(next_states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        self.augment_prediction(prediction)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        target_atoms = self.atoms.view(1, -1)
        target_prob = prediction['v_prob'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            target_atoms = storage.r[i] + config.discount * storage.m[i] * target_atoms
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()
            storage.t_atoms[i] = target_atoms
            storage.t_probs[i] = target_prob

        states, actions, log_probs_old, returns, advantages, next_states, rewards, masks, target_atoms, target_probs\
            = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv', 'next_s', 'r', 'm', 't_atoms', 't_probs'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        if config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                sampled_next_states = next_states[batch_indices]
                sampled_rewards = rewards[batch_indices]
                sampled_masks = masks[batch_indices]
                sampled_target_atoms = target_atoms[batch_indices]
                sampled_target_probs = target_probs[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                if config.critic_update == 'td':
                    with torch.no_grad():
                        next_prediction = self.network(sampled_next_states)
                    atoms_target = sampled_rewards + config.discount * sampled_masks * self.atoms.view(1, -1)
                    atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
                    next_prob = next_prediction['v_prob']
                elif config.critic_update == 'mc':
                    atoms_target = sampled_target_atoms
                    next_prob = sampled_target_probs
                atoms_target = atoms_target.unsqueeze(1)
                target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * next_prob.unsqueeze(1)
                target_prob = target_prob.sum(-1)

                value_loss = -(target_prob * prediction['v_log_prob']).sum(-1).mean()
                if config.aux:
                    canonical_v_loss = (sampled_returns - prediction['v_canonical']).pow(2).mul(0.5).mean()
                    value_loss = value_loss + canonical_v_loss

                approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()
                if config.shared_repr:
                    self.opt.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.opt.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()

