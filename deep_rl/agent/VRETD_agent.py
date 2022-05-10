#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import collections

import torch

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class VRETDAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        def lr_fn(epoch):
            return 1
            # return min(1.0 / (epoch + 1), 0.00001)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lr_fn)
        self.total_steps = 0
        self.state = None
        if config.n >= 0:
            self.rhos = collections.deque(maxlen=config.n)
        else:
            self.rhos = None
        self.F = 1

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        prob_pi = torch.softmax(q / self.config.tau, dim=-1)
        dist = torch.distributions.Categorical(probs=prob_pi)
        action = to_np(dist.sample())
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        if self.state is None:
            self.state = self.task.reset()

        env = self.task.env.envs[0].env
        if 'Baird' in config.game:
            if config.game == 'BairdPrediction-v0':
                action_info = env.act(pi_dashed=config.pi_dashed)
            elif config.game == 'BairdControl-v0':
                q = self.network(env.expand_phi(self.state[0])).flatten()
                prob = torch.softmax(q, dim=-1)
                prob = to_np(prob)
                prob_pi = torch.softmax(q / config.tau, dim=-1)
                prob_pi = to_np(prob_pi)
                pi_dashed = prob_pi[0]
                if config.softmax_mu:
                    epsilon = 0.9
                    action_info = env.act(epsilon * 6.0 / 7 + (1 - epsilon) * prob[0], pi_dashed=pi_dashed)
                else:
                    action_info = env.act(pi_dashed=pi_dashed)
            else:
                raise NotImplementedError
            action = action_info['action']
        else:
            q = self.network(self.state)['q']
            prob_pi = torch.softmax(q / config.tau, dim=-1)
            prob_mu = torch.softmax(q, dim=-1)
            random_mu = torch.softmax(q * 0, dim=-1)
            prob_mu = config.eps * random_mu + (1 - config.eps) * prob_mu
            dist = torch.distributions.Categorical(probs=prob_mu)
            action = dist.sample().unsqueeze(-1)
            action_info = dict(
                pi_prob=np.asscalar(to_np(prob_pi.gather(1, action))),
                mu_prob=np.asscalar(to_np(prob_mu.gather(1, action))), 
            )
            action = to_np(action)[0, 0]

        next_state, reward, done, info = self.task.step([action])
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)
        reward = tensor(reward).view(1, -1)

        if config.game == 'BairdPrediction-v0':
            target = reward + config.discount * self.network(next_state)
            target = target.detach()
            rho_t = action_info['pi_prob'] / action_info['mu_prob']
            if config.n < 0:
                F = self.F
            else:
                F = 1
                for rho in self.rhos:
                    F = 1 + config.beta * rho * F
                self.rhos.append(rho_t)
            loss = (target - self.network(self.state)).pow(2).mul(0.5).mul(rho_t * F).mean()
            if config.n < 0:
                self.F = 1 + config.beta * rho_t * self.F
        elif config.game in ['BairdControl-v0', 'CartPole-v0', 'Acrobot-v1']:
            rho_t = action_info['pi_prob'] / action_info['mu_prob']
            if config.n < 0:
                F = 1 + config.beta * rho_t * self.F
            else:
                self.rhos.append(rho_t)
                F = 1
                for rho in self.rhos:
                    F = 1 + config.beta * rho * F
            if config.game == 'BairdControl-v0':
                q_next = self.network(env.expand_phi(next_state[0]))
                pi_next = torch.softmax(q_next / config.tau, dim=0)
                target = (q_next * pi_next).sum(0, keepdim=True)
                q = self.network(env.expand_phi(self.state[0]))
                q = q[[action]]
            else:
                q = self.network(self.state)['q']
                q = q[0, action]
                q_next = self.network(next_state)['q']
                pi_next = torch.softmax(q_next / config.tau, dim=-1)
                target = (q_next * pi_next).sum(-1, keepdim=True)
            mask = tensor(1 - done).unsqueeze(-1)
            target = reward + mask * config.discount * target
            target = target.detach()
            loss = (target - q).pow(2).mul(0.5 * F).mean()
            if config.n < 0:
                self.F = F
        else:
            raise NotImplementedError

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.lr_scheduler.step()

        self.state = next_state
        self.total_steps += 1

    def eval_episodes(self):
        config = self.config
        if 'Baird' not in config.game:
            super().eval_episodes()
            return
        env = self.task.env.envs[0].env
        # for i in range(config.state_dim):
        #     self.logger.add_scalar('weight %d' % (i), self.network.fc.weight.data.flatten()[i])
        self.logger.add_scalar('weight', self.network.fc.weight.data.flatten().norm())
        if config.game == 'BairdPrediction-v0':
            phi = env.phi
        elif config.game == 'BairdControl-v0':
            phi = np.concatenate([env.expand_phi(phi) for phi in env.phi], axis=0)
        else:
            raise NotImplementedError
        error = self.network(phi).flatten().norm()
        print(error)
        self.logger.add_scalar('error', error)
