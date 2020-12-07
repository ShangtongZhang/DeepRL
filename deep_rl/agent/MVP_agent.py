#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class MVPWorker:
    def __init__(self, config, master_network):
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
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

    def compute_y_grad(self):
        config = self.config
        y = self.network.y
        y_loss = (2 * self.total_rewards + 1.0 / config.lam) * y - y.pow(2)
        y_loss = -y_loss
        self.opt.zero_grad()
        y_loss.backward()

    def compute_pi_grad(self):
        config = self.config
        R = self.total_rewards
        y = self.network.y
        policy_loss = -(2 * y.detach() * R - R.pow(2)) * self.log_pi_a
        policy_loss = policy_loss - config.entropy_weight * self.entropy
        self.opt.zero_grad()
        policy_loss.backward()

    def post_step(self, reward, terminal):
        self.total_rewards += reward


class MVPAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.workers = [MVPWorker(config, self.network) for _ in range(config.num_workers)]
        self.opt = config.optimizer_fn(self.network.parameters())
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
                worker.compute_y_grad()
                self.opt.zero_grad()
                sync_grad(self.network, worker.network)
                self.opt.step()
                worker.sync()
                worker.compute_pi_grad()
                self.opt.zero_grad()
                sync_grad(self.network, worker.network)
                self.opt.step()
                worker.reset()

        for i, worker in enumerate(self.workers):
            if terminals[i]:
                worker.sync()

        self.states = next_states
        self.total_steps += config.num_workers