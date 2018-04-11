#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicNet:
    def __init__(self, gpu):
        if not torch.cuda.is_available():
            gpu = -1
        self.gpu = gpu
        if self.gpu >= 0:
            self.cuda(self.gpu)

    def supported_dtype(self, x, torch_type):
        if torch_type == torch.FloatTensor:
            return np.asarray(x, dtype=np.float32)
        if torch_type == torch.LongTensor:
            return np.asarray(x, dtype=np.int64)

    def variable(self, x, dtype=torch.FloatTensor):
        if isinstance(x, Variable):
            return x
        x = dtype(torch.from_numpy(self.supported_dtype(x, dtype)))
        if self.gpu >= 0:
            x = x.cuda(self.gpu)
        return Variable(x)

    def tensor(self, x, dtype=torch.FloatTensor):
        x = dtype(torch.from_numpy(self.supported_dtype(x, dtype)))
        if self.gpu >= 0:
            x = x.cuda(self.gpu)
        return x

class VanillaNet(BasicNet):
    def __init__(self, feature_dim, output_dim, gpu):
        self.fc_head = nn.Linear(feature_dim, output_dim)
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.feature(x)
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y

class DuelingNet(BasicNet):
    def __init__(self, feature_dim, action_dim, gpu):
        self.fc_value = nn.Linear(feature_dim, 1)
        self.fc_advantage = nn.Linear(feature_dim, action_dim)
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.feature(x)
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().data.numpy()
        return q

class ActorCriticNet(BasicNet):
    def __init__(self, feature_dim, action_dim, gpu):
        self.fc_actor = nn.Linear(feature_dim, action_dim)
        self.fc_critic = nn.Linear(feature_dim, 1)
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.feature(x)
        pre_prob = self.fc_actor(phi)
        prob = F.softmax(pre_prob, dim=1)
        log_prob = F.log_softmax(pre_prob, dim=1)
        value = self.fc_critic(phi)
        if to_numpy:
            return prob.cpu().data.numpy()
        return prob, log_prob, value

class CategoricalNet(BasicNet):
    def __init__(self, feature_dim, action_dim, num_atoms, gpu):
        self.fc_categorical = nn.Linear(feature_dim, action_dim * num_atoms)
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.feature(x)
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().data.numpy()
        return prob

class QuantileNet(BasicNet):
    def __init__(self, feature_dim, action_dim, num_quantiles, gpu):
        self.fc_quantiles = nn.Linear(feature_dim, action_dim * num_quantiles)
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.feature(x)
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.data.cpu().numpy()
        return quantiles

class NatureConvNet(nn.Module):
    def __init__(self, in_channels):
        super(NatureConvNet, self).__init__()
        self.feature_dim = 512
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, self.feature_dim)

        for layer in self.children():
            relu_gain = nn.init.calculate_gain('relu')
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal(layer.weight.data, relu_gain)
            nn.init.constant(layer.bias.data, 0)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class TwoLayerFCNet(nn.Module):
    def __init__(self, state_dim, hidden_size=64, gate=F.relu):
        super(TwoLayerFCNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = gate

    def forward(self, x):
        y = self.gate(self.fc1(x))
        y = self.gate(self.fc2(y))
        return y

class ContinuousActorCriticWrapper:
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

    def variable(self, x, dtype=torch.FloatTensor):
        return self.actor.variable(x, dtype)

    def tensor(self, x, dtype=torch.FloatTensor):
        return self.actor.tensor(x, dtype)

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

class DiscreteActorCriticWrapper:
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
        return action, log_prob, entropy_loss, value

    def variable(self, x, dtype=torch.FloatTensor):
        return self.network.variable(x, dtype)

    def tensor(self, x, dtype=torch.FloatTensor):
        return self.network.tensor(x, dtype)

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
