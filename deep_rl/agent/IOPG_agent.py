#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class IOPGWorker:
    def __init__(self, config, master_network):
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.config = config
        self.master_network = master_network

        self.worker_index = tensor([0]).long()
        self.reset()

    def sync(self):
        self.network.load_state_dict(self.master_network.state_dict())

    def reset(self):
        config = self.config
        self.m = tensor(np.ones((1, config.num_o))) / config.num_o
        self.is_initial_states = tensor(np.ones((1))).byte()
        self.prev_options = tensor(np.zeros(1)).long()
        self.storage = Storage(int(1e6))

    def compute_pi_hat(self, prediction, prev_option, is_intial_states):
        inter_pi = prediction['inter_pi']
        mask = torch.zeros_like(inter_pi)
        mask[self.worker_index, prev_option] = 1
        beta = prediction['beta']
        pi_hat = (1 - beta) * mask + beta * inter_pi
        is_intial_states = is_intial_states.view(-1, 1).expand(-1, inter_pi.size(1))
        pi_hat = torch.where(is_intial_states, inter_pi, pi_hat)
        return pi_hat

    def pre_step(self, state):
        config = self.config
        prediction = self.network(state)
        pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
        dist = torch.distributions.Categorical(probs=pi_hat)
        options = dist.sample()

        mean = prediction['mean'][self.worker_index, options]
        std = prediction['std'][self.worker_index, options]
        dist = torch.distributions.Normal(mean, std)
        actions = dist.sample()

        self.storage.add(prediction)

        dist = torch.distributions.Normal(prediction['mean'], prediction['std'])
        a = actions.unsqueeze(1).expand(-1, pi_hat.size(1), -1)
        pi_o_a = dist.log_prob(a).sum(-1).exp()
        m_pi_o_a = self.m * pi_o_a
        c = m_pi_o_a.sum(-1).unsqueeze(-1).pow(-1)

        pi_hat = []
        for i in range(config.num_o):
            p = self.compute_pi_hat(prediction, tensor([i]).long(), self.is_initial_states)
            pi_hat.append(p.unsqueeze(1))
        pi_hat = torch.cat(pi_hat, dim=1)
        self.m = c.unsqueeze(-1) * m_pi_o_a.unsqueeze(-1) * pi_hat
        self.m = self.m.sum(1)

        pre_log = (self.m * pi_o_a).sum(-1).unsqueeze(-1)
        self.storage.add({'pre_log': pre_log})

        self.prev_options = options
        return actions

    def learn(self):
        config = self.config
        storage = self.storage
        storage.size = len(storage.r)
        storage.placeholder()
        ret = tensor([[0]])
        for i in reversed(range(len(storage.r))):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            storage.ret[i] = ret

        pre_log, q_o, ret = storage.cat(['pre_log', 'q_o', 'ret'])
        v = q_o[:, [0]]
        adv = ret - v

        policy_loss = pre_log.add(1e-5).log() * adv.detach()
        policy_loss = -policy_loss.mean()
        v_loss = adv.pow(2).mul(0.5).mean()

        self.opt.zero_grad()
        (policy_loss + v_loss).backward()

    def post_step(self, reward, terminal):
        self.storage.add(dict(r=tensor(reward).view(1, 1),
                              m=1 - tensor(terminal).view(1, 1)))
        self.is_initial_states = self.storage.m[-1].byte().view(-1)
        if terminal:
            self.learn()
            self.reset()


class IOPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.workers = [IOPGWorker(config, self.network) for _ in range(config.num_workers)]
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0

        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

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
                self.opt.zero_grad()
                sync_grad(self.network, worker.network)
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

        for i, worker in enumerate(self.workers):
            if terminals[i]:
                worker.sync()

        self.states = next_states
        self.total_steps += config.num_workers
