#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseNet:
    def set_gpu(self, gpu):
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:%d' % (gpu))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def tensor(self, x):
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return x

class DisjointActorCriticWrapper:
    def __init__(self, state_dim, action_dim, actor_network_fn, critic_network_fn):
        self.actor = actor_network_fn(state_dim, action_dim)
        self.critic = critic_network_fn(state_dim, action_dim)

    def state_dict(self):
        return [self.actor.state_dict(), self.critic.state_dict()]

    def load_state_dict(self, state_dicts):
        self.actor.load_state_dict(state_dicts[0])
        self.critic.load_state_dict(state_dicts[1])

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def zero_grad(self):
        self.actor.zero_grad()
        self.critic.zero_grad()

class GaussianActorCriticWrapper:
    def __init__(self, state_dim, action_dim, actor_fn, critic_fn, actor_opt_fn, critic_opt_fn):
        self.actor = actor_fn(state_dim, action_dim)
        self.critic = critic_fn(state_dim)
        self.actor_opt = actor_opt_fn(self.actor.parameters())
        self.critic_opt = critic_opt_fn(self.critic.parameters())

    def predict(self, state, actions=None):
        mean, std, log_std = self.actor.predict(state)
        values = self.critic.predict(state)
        dist = torch.distributions.Normal(mean, std)
        if actions is None:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=1, keepdim=True)
        return actions, log_probs, 0, values

    def tensor(self, x):
        return self.actor.tensor(x)

    def zero_grad(self):
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def step(self):
        self.actor_opt.step()
        self.critic_opt.step()

    def state_dict(self):
        return [self.actor.state_dict(), self.critic.state_dict()]

    def load_state_dict(self, state_dicts):
        self.actor.load_state_dict(state_dicts[0])
        self.critic.load_state_dict(state_dicts[1])

class CategoricalActorCriticWrapper:
    def __init__(self, state_dim, action_dim, network_fn, opt_fn):
        self.network = network_fn(state_dim, action_dim)
        self.opt = opt_fn(self.network.parameters())

    def predict(self, state, action=None):
        prob, log_prob, value = self.network.predict(state)
        entropy_loss = torch.sum(prob * log_prob, dim=1, keepdim=True)
        dist = torch.distributions.Categorical(prob)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(1)
        return action, log_prob, entropy_loss.mean(0), value

    def tensor(self, x):
        return self.network.tensor(x)

    def zero_grad(self):
        self.opt.zero_grad()

    def parameters(self):
        return self.network.parameters()

    def step(self):
        self.opt.step()

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state_dicts):
        self.network.load_state_dict(state_dicts)

def layer_init(layer):
    nn.init.orthogonal_(layer.weight.data)
    nn.init.constant_(layer.bias.data, 0)
    return layer