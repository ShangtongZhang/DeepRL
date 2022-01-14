#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torch.multiprocessing as mp


def rollout_from(states, network, discount, normalizer, env):
    all_rewards = []
    while True:
        prediction = network(states)
        next_states, rewards, terminals, info = env.step(
            to_np(prediction['a']))
        all_rewards.append(rewards[0])
        if terminals[0]:
            break
        states = normalizer(next_states)
    cum_r = 0
    for r in reversed(all_rewards):
        cum_r = r + discount * cum_r
    return cum_r


class MonteCarloEstimator(mp.Process):
    INIT = 1
    ROLLOUT = 2

    def __init__(self, game, discount):
        mp.Process.__init__(self)
        self.game = game
        self.discount = discount
        self.pipe, self.worker_pipe = mp.Pipe()

    def run(self):
        env = None
        net = None
        while True:
            op, data = self.worker_pipe.recv()
            if op == self.INIT:
                env = Task(self.game)
                env.reset()
                net = data
            elif op == self.ROLLOUT:
                states, mj_state, normalizer = data
                env.reset()
                env.set_state(mj_state)
                ret = rollout_from(states, net, self.discount, normalizer, env)
                self.worker_pipe.send(ret)

    def init(self, net):
        self.pipe.send([self.INIT, net])

    def rollout(self, states, mj_state, normalizer):
        self.pipe.send([self.ROLLOUT, [states, mj_state, normalizer]])
        return self.pipe.recv()


class MCEstimators:
    def __init__(self, config):
        self.ps = [MonteCarloEstimator(
            config.game, config.discount) for _ in range(config.mc_n)]
        for p in self.ps:
            p.start()

    def init(self, net):
        for p in self.ps:
            p.init(net)

    def rollout(self, states, mj_state, normalizer):
        rets = [p.rollout(states, mj_state, normalizer) for p in self.ps]
        return rets


class MPPPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.oracle = config.task_fn()
        self.oracle.reset()
        self.network = config.network_fn()
        self.network.share_memory()
        if config.shared_repr:
            self.opt = config.optimizer_fn(self.network.parameters())
        else:
            self.actor_opt = config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        if config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lambda step: 1 - step / config.max_steps)
        self.MCEstimator = MCEstimators(config)
        self.MCEstimator.init(self.network)

    def sample_trajectory_from(self, states, mj_state):
        config = self.config
        env = self.oracle
        env.reset()
        env.set_state(mj_state)
        all_rewards = []
        while True:
            prediction = self.network(states)
            next_states, rewards, terminals, info = env.step(
                to_np(prediction['a']))
            all_rewards.append(rewards[0])
            if terminals[0]:
                break
            states = config.state_normalizer(next_states)
        cum_r = 0
        for r in reversed(all_rewards):
            cum_r = r + config.discount * cum_r
        return cum_r

    def compute_oracle_v(self, states, mj_state):
        self.config.state_normalizer.set_read_only()
        rets = self.MCEstimator.rollout(
            states, mj_state, self.config.state_normalizer)
        self.config.state_normalizer.unset_read_only()
        return tensor(rets).mean().view(1, 1)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['oracle_v', 'oracle_adv'])
        states = self.states
        for _ in range(config.rollout_length):
            if config.use_oracle_v:
                oracle_v = self.compute_oracle_v(states, self.task.get_state())
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(
                to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            if config.use_oracle_v:
                prediction['v'] = oracle_v
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'next_s': tensor(next_states),
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        if config.use_oracle_v:
            oracle_v = self.compute_oracle_v(states, self.task.get_state())
            prediction['v'] = oracle_v
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * \
                    storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * \
                    config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages, next_states, rewards, masks \
            = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv', 'next_s', 'r', 'm'])

        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        if config.normalized_adv:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

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

                sampled_next_states = next_states[batch_indices]
                sampled_rewards = rewards[batch_indices]
                sampled_masks = masks[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - \
                    config.entropy_weight * prediction['ent'].mean()

                if config.critic_update == 'mc':
                    value_loss = 0.5 * (sampled_returns -
                                        prediction['v']).pow(2).mean()
                elif config.critic_update == 'td':
                    with torch.no_grad():
                        prediction_next = self.network(sampled_next_states)
                    target = sampled_rewards + config.discount * \
                        sampled_masks + prediction_next['v']
                    value_loss = 0.5 * (target - prediction['v']).pow(2).mean()

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
