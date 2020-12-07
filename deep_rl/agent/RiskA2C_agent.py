#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class RiskA2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.network_plus = config.network_fn()
        self.network_plus.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(list(self.network.parameters()) +
                                             list(self.network_plus.parameters()))
        self.total_steps = 0
        self.states = self.task.reset()

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(prediction['a'])

    def step(self):
        config = self.config
        noise = dict()
        for name, param in self.network_plus.actor_named_parameters.items():
            param.data.copy_(self.network.actor_named_parameters[name])
            noise[name] = torch.randn(param.size())
            param.data.add_(config.beta * noise[name])

        sep = config.num_workers // 2
        states = self.states[:sep]
        states_plus = self.states[sep:]
        prediction = self.network(states)
        prediction_plus = self.network_plus(states_plus)
        actions = torch.cat([prediction['a'], prediction_plus['a']], dim=0)
        next_states, rewards, terminals, info = self.task.step(to_np(actions))
        mask = tensor(1 - terminals).unsqueeze(-1)
        self.record_online_return(info)
        rewards = config.reward_normalizer(rewards)
        rewards = tensor(rewards).unsqueeze(-1)

        self.states = next_states
        next_states = self.states[:sep]
        next_states_plus = self.states[sep:]

        with torch.no_grad():
            prediction_next = self.network(next_states)
            prediction_next_plus = self.network_plus(next_states_plus)

        v = torch.cat([prediction['v'], prediction_plus['v']], dim=0)
        v_next = torch.cat([prediction_next['v'], prediction_next_plus['v']], dim=0)
        v_loss = (rewards + config.discount * mask * v_next - v).pow(2).mul(0.5).mean()

        u = torch.cat([prediction['u'], prediction_plus['u']], dim=0)
        u_next = torch.cat([prediction_next['u'], prediction_next_plus['u']], dim=0)
        u_loss = (rewards.pow(2) + 2 * config.discount * rewards * mask * v_next +
                  config.discount ** 2 * mask * u_next - u).pow(2).mul(0.5).mean()

        initial_states = [config.eval_env.reset() for _ in range(config.num_workers)]
        initial_states = tensor(initial_states).squeeze(1)
        with torch.no_grad():
            prediction = self.network(initial_states)
            prediction_plus = self.network_plus(initial_states)
            grad = (1 + 2 * config.lam * prediction['v']) * (prediction_plus['v'] - prediction['v']) - \
                config.lam * (prediction_plus['u'] - prediction['u'])
            grad = config.pi_loss_weight * grad.mean()

        entropy_loss = -prediction['ent'].mean() * config.pi_loss_weight
        self.optimizer.zero_grad()
        for name, param in self.network.actor_named_parameters.items():
            param.grad = -noise[name] * grad / config.beta
        (config.entropy_weight * entropy_loss + v_loss + u_loss).backward()
        # nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
        self.total_steps += config.num_workers
