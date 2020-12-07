#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class TamarWorker:
    def __init__(self, config, master_network):
        self.network = config.network_fn()
        self.config = config
        self.master_network = master_network

        self.reset()

    def sync(self):
        self.network.load_state_dict(self.master_network.state_dict())

    def reset(self):
        self.total_rewards = tensor(0)
        self.log_pi_a = 0
        self.entropy = 0

    def pre_step(self, state):
        prediction = self.network(state)
        self.log_pi_a = self.log_pi_a + prediction['log_pi_a']
        self.entropy = self.entropy + prediction['ent']
        return prediction['a']

    def compute_loss(self):
        config = self.config
        R = self.total_rewards
        J_loss = (R - self.network.J).pow(2).mul(0.5)
        V_loss = (R.pow(2) - self.network.J.detach().pow(2) - self.network.V).pow(2).mul(0.5)
        if np.asscalar(to_np(self.network.V)) < config.b:
            grad = 0
        else:
            grad = 2 * (self.network.V.detach() - config.b)
        policy_loss = -(R - config.lam * grad * (R.pow(2) - 2 * self.network.J.detach().pow(2))) * self.log_pi_a
        policy_loss = policy_loss - config.entropy_weight * self.entropy
        self.network.zero_grad()
        (J_loss + V_loss + policy_loss).backward()

    def post_step(self, reward, terminal):
        self.total_rewards += reward


class TamarAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.workers = [TamarWorker(config, self.network) for _ in range(config.num_workers)]
        self.opt_pi = config.pi_optimizer_fn(self.network.pi_params)
        self.opt_JV = config.JV_optimizer_fn(self.network.JV_params)
        self.total_steps = 0

        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(prediction['a'])

    def step(self):
        config = self.config
        actions = []
        for i, worker in enumerate(self.workers):
            actions.append(worker.pre_step(self.states[[i]]))
        actions = torch.cat(actions, dim=0)
        next_states, rewards, terminals, info = self.task.step(to_np(actions))
        self.record_online_return(info)
        rewards = config.reward_normalizer(rewards)
        next_states = config.state_normalizer(next_states)

        for i, worker in enumerate(self.workers):
            worker.post_step(rewards[i], terminals[i])
            if terminals[i]:
                worker.compute_loss()
                self.opt_JV.zero_grad()
                self.opt_pi.zero_grad()
                sync_grad(self.network, worker.network)
                self.opt_JV.step()
                self.opt_pi.step()
                worker.reset()

        for i, worker in enumerate(self.workers):
            if terminals[i]:
                worker.sync()

        self.states = next_states
        self.total_steps += config.num_workers