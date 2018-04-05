#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .base_network import *

class DeterministicActorNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_gate=F.tanh,
                 action_scale=1,
                 gpu=-1,
                 non_linear=F.tanh,
                 hidden_size=64):
        super(DeterministicActorNet, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_dim)
        self.action_gate = action_gate
        self.action_scale = action_scale
        self.non_linear = non_linear
        self.init_weights()
        BasicNet.__init__(self, gpu)

    def init_weights(self):
        bound = 3e-3
        nn.init.uniform(self.layer3.weight.data, -bound, bound)
        nn.init.constant(self.layer3.bias.data, 0)

        nn.init.xavier_uniform(self.layer1.weight.data)
        nn.init.constant(self.layer1.bias.data, 0)
        nn.init.xavier_uniform(self.layer2.weight.data)
        nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x):
        x = self.variable(x)
        x = self.non_linear(self.layer1(x))
        x = self.non_linear(self.layer2(x))
        x = self.layer3(x)
        x = self.action_scale * self.action_gate(x)
        return x

    def predict(self, x, to_numpy=False):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y

class DeterministicCriticNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 gpu=-1,
                 non_linear=F.tanh,
                 hidden_size=64):
        super(DeterministicCriticNet, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.non_linear = non_linear
        self.init_weights()
        BasicNet.__init__(self, gpu)

    def init_weights(self):
        bound = 3e-3
        nn.init.uniform(self.layer3.weight.data, -bound, bound)
        nn.init.constant(self.layer3.bias.data, 0)

        nn.init.xavier_uniform(self.layer1.weight.data)
        nn.init.constant(self.layer1.bias.data, 0)
        nn.init.xavier_uniform(self.layer2.weight.data)
        nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x, action):
        x = self.variable(x)
        action = self.variable(action)
        x = self.non_linear(self.layer1(x))
        x = self.non_linear(self.layer2(torch.cat([x, action], dim=1)))
        x = self.layer3(x)
        return x

    def predict(self, x, action):
        return self.forward(x, action)

class GaussianActorNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_scale=1,
                 action_gate=F.tanh,
                 gpu=-1,
                 unit_std=True,
                 hidden_size=64,
                 non_linear=F.tanh):
        super(GaussianActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_mean = nn.Linear(hidden_size, action_dim)

        if unit_std:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.action_std = nn.Linear(hidden_size, action_dim)

        self.unit_std = unit_std
        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        phi = self.non_linear(self.fc1(x))
        phi = self.non_linear(self.fc2(phi))
        mean = self.action_mean(phi)
        if self.action_gate is not None:
            mean = self.action_scale * self.action_gate(mean)
        if self.unit_std:
            log_std = self.action_log_std.expand_as(mean)
            std = log_std.exp()
        else:
            std = F.softplus(self.action_std(phi)) + 1e-5
            log_std = std.log()
        return mean, std, log_std

    def predict(self, x):
        return self.forward(x)

    def log_density(self, x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * var + 1e-5) - 0.5 * torch.log(2 * Variable(torch.FloatTensor([np.pi])).expand_as(x)) - log_std
        return log_density.sum(1)

    def entropy(self, std):
        return 0.5 * (1 + (2 * std.pow(2) * np.pi + 1e-5).log()).sum(1).mean()

class GaussianCriticNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 gpu=-1,
                 hidden_size=64,
                 non_linear=F.tanh):
        super(GaussianCriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.non_linear = non_linear
        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        phi = self.non_linear(self.fc1(x))
        phi = self.non_linear(self.fc2(phi))
        value = self.fc_value(phi)
        return value

    def predict(self, x):
        return self.forward(x)

class DisjointActorCriticNet:
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

    def train(self):
        self.actor.train()
        self.critic.train()
