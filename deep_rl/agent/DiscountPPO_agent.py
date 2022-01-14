#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class DiscountPPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.task_extra = config.task_fn()
        self.network = config.network_fn()
        if config.use_target_net:
            self.target_net = config.network_fn()
            self.target_net.load_state_dict(self.network.state_dict())
        if config.shared_repr:
            self.opt = config.optimizer_fn(self.network.parameters())
        else:
            self.actor_opt = config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.gammas = tensor(np.ones((self.states.shape[0], 1)))
        self.timesteps = tensor(np.ones((self.states.shape[0], 1)))
        if config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lambda step: 1 - step / config.max_steps)
        if config.multi_path:
            self.oracle = config.task_fn()
            self.oracle.reset()

    def rollout_oracle(self, mj_state, action_dist):
        config = self.config
        config.state_normalizer.set_read_only()
        next_states, rewards, masks = [], [], []
        for i in range(config.multi_path):
            self.oracle.reset()
            self.oracle.set_state(mj_state)
            action = action_dist.sample()
            next_state, reward, terminal, _ = self.oracle.step(to_np(action))
            next_states.append(config.state_normalizer(next_state))
            rewards.append(reward)
            masks.append(1 - np.asarray(terminal))
        config.state_normalizer.unset_read_only()
        next_states = tensor(next_states).squeeze(1)
        rewards = tensor(rewards)
        masks = tensor(masks)
        return dict(mp_next_s=next_states,
                    mp_r=rewards,
                    mp_m=masks)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        if self.use_aux_action:
            action = prediction['a_aux']
        else:
            action = prediction['a']
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def eval_episodes(self):
        self.field = 'main'
        self.use_aux_action = False
        super().eval_episodes()
        if self.config.aux:
            self.field = 'aux'
            self.use_aux_action = True
            super().eval_episodes()

    def collect_extra_data(self):
        config = self.config
        self.config.state_normalizer.set_read_only()
        states = self.task_extra.reset()
        states = self.config.state_normalizer(states)
        storage = Storage(config.extra_data * config.rollout_length)
        for _ in range(config.extra_data * config.rollout_length):
            prediction = self.network(states)
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
        storage.add(prediction)
        self.config.state_normalizer.unset_read_only()
        return storage

    def train_with_extra_data(self, storage):
        config = self.config
        storage.placeholder()

        returns = storage.v[-1].detach()
        for i in reversed(range(config.extra_data * config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            storage.ret[i] = returns.detach()

        states, returns, next_states, rewards, masks = storage.cat(['s', 'ret', 'next_s', 'r', 'm'])
        for _ in range(config.optimization_epochs):
            sampler = random_sample(
                np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_returns = returns[batch_indices]

                sampled_next_states = next_states[batch_indices]
                sampled_rewards = rewards[batch_indices]
                sampled_masks = masks[batch_indices]

                prediction = self.network(sampled_states)
                if config.critic_update == 'mc':
                    value_loss = 0.5 * (sampled_returns -
                                        prediction['v']).pow(2).mean()
                elif config.critic_update == 'td':
                    with torch.no_grad():
                        prediction_next = self.network(sampled_next_states)
                    target = sampled_rewards + config.discount * \
                             sampled_masks * prediction_next['v']
                    value_loss = 0.5 * (target - prediction['v']).pow(2).mean()

                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()

    def augment_prediction(self, prediction, mj_state, next_states, rewards, masks):
        mp_data = self.rollout_oracle(mj_state, prediction['action_dist'])
        mp_data['mp_next_s'] = torch.cat([mp_data['mp_next_s'], next_states], dim=0).unsqueeze(0)
        mp_data['mp_r'] = torch.cat([mp_data['mp_r'], rewards], dim=0).unsqueeze(0)
        mp_data['mp_m'] = torch.cat([mp_data['mp_m'], masks], dim=0).unsqueeze(0)
        del prediction['action_dist']
        prediction.update(mp_data)

    def step(self):
        config = self.config
        if config.extra_data:
            storage = self.collect_extra_data()
            self.train_with_extra_data(storage)

        storage = Storage(config.rollout_length, ['gamma', 't', 'mp_next_s', 'mp_r', 'mp_m'])
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(
                to_np(prediction['a']))
            self.record_online_return(info)
            if config.flip_r:
                rewards = tensor(rewards)
                rewards = torch.where(self.timesteps.view(-1) < config.flip_r, rewards, -rewards)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)
            if config.multi_path:
                self.augment_prediction(prediction, self.task.get_state(), tensor(next_states),
                                        tensor(rewards).unsqueeze(-1), mask)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': mask,
                         's': tensor(states),
                         'gamma': self.gammas,
                         't': self.timesteps,
                         'next_s': tensor(next_states),
                         })
            states = next_states
            self.total_steps += config.num_workers
            self.gammas = self.gammas * config.discount
            self.gammas = torch.where(
                mask == 1, self.gammas, torch.ones_like(self.gammas))
            self.timesteps = self.timesteps + 1
            self.timesteps = torch.where(mask == 1, self.timesteps, torch.zeros_like(self.timesteps))

        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.adv_gamma * \
                           storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * \
                             config.adv_gamma * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages, gammas, timesteps, next_states, rewards, masks = storage.cat(
            ['s', 'a', 'log_pi_a', 'ret', 'adv', 'gamma', 't', 'next_s', 'r', 'm'])
        if config.multi_path:
            mp_next_states, mp_rewards, mp_masks = storage.cat(['mp_next_s', 'mp_r', 'mp_m'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()
        if config.d_scheme == Config.NO:
            gammas = gammas.mul(0).add(1)
        elif config.d_scheme == Config.UNBIAS:
            pass
        elif config.d_scheme == Config.COMP_UNBIAS:
            gammas = 1 - gammas * config.discount
        elif config.d_scheme == Config.INV_LINEAR:
            timesteps = (timesteps + 1) ** -1
        elif config.d_scheme == Config.LOG_LINEAR:
            timesteps = timesteps.add(1).log()
        else:
            raise NotImplementedError

        if config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

        if config.sync_aux:
            self.network.sync_aux()

        v_max = 0
        for _ in range(config.optimization_epochs):
            sampler = random_sample(
                np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                sampled_gammas = gammas[batch_indices]
                sampled_ts = timesteps[batch_indices]

                sampled_next_states = next_states[batch_indices]
                sampled_rewards = rewards[batch_indices]
                sampled_masks = masks[batch_indices]

                if config.d_scheme in [Config.NO, Config.UNBIAS, Config.COMP_UNBIAS]:
                    state_discounting = sampled_gammas
                elif config.d_scheme in [Config.INV_LINEAR, Config.LOG_LINEAR]:
                    state_discounting = sampled_ts
                else:
                    raise NotImplementedError

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mul(state_discounting).mean() - \
                              config.entropy_weight * prediction['ent'].mean()

                if config.aux:
                    state_discounting = 1 - sampled_gammas
                    ratio_aux = (prediction['log_pi_a_aux'] -
                                 sampled_log_probs_old).exp()
                    obj = ratio_aux * sampled_advantages
                    obj_clipped = ratio_aux.clamp(1.0 - self.config.ppo_ratio_clip,
                                                  1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                    aux_loss = -torch.min(obj, obj_clipped).mul(state_discounting).mean()
                    policy_loss = policy_loss + aux_loss

                if config.critic_update == 'mc':
                    value_loss = 0.5 * (sampled_returns -
                                        prediction['v']).pow(2).mean()
                elif config.critic_update == 'td':
                    network = self.target_net if config.use_target_net else self.network
                    if config.multi_path:
                        mp_next_s = mp_next_states[batch_indices]
                        with torch.no_grad():
                            prediction_next = network(mp_next_s)
                            target = mp_rewards[batch_indices] + config.discount * mp_masks[batch_indices] * \
                                     prediction_next['v']
                            target = target.mean(1)
                    else:
                        with torch.no_grad():
                            prediction_next = network(sampled_next_states)
                        target = sampled_rewards + config.discount * \
                                 sampled_masks * prediction_next['v']
                    value_loss = 0.5 * (target - prediction['v']).pow(2).mean()

                v_max = max(v_max, np.asscalar(to_np(prediction['v'].max())))

                approx_kl = (sampled_log_probs_old -
                             prediction['log_pi_a']).mean()
                if config.shared_repr:
                    self.opt.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), config.gradient_clip)
                    self.opt.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()

        if config.use_target_net:
            self.target_net.load_state_dict(self.network.state_dict())