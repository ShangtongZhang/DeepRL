#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *

class DDPGActorNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_gate,
                 action_scale,
                 gpu=False):
        super(DDPGActorNet, self).__init__()
        hidden1 = 400
        hidden2 = 300
        self.layer1 = nn.Linear(state_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.layer3 = nn.Linear(hidden2, action_dim)
        self.action_gate = action_gate
        self.action_scale = action_scale
        BasicNet.__init__(self, None, False, False)
        self.init_weights()

    def init_weights(self):
        bound = 3e-3
        self.layer3.weight.data.uniform_(-bound, bound)
        # self.layer3.bias.data.uniform_(-bound, bound)

        def fanin(size):
            v = 1.0 / np.sqrt(size[1])
            return torch.FloatTensor(size).uniform_(-v, v)

        self.layer1.weight.data = fanin(self.layer1.weight.data.size())
        # self.layer1.bias.data = fanin(self.layer1.bias.data.size())
        self.layer2.weight.data = fanin(self.layer2.weight.data.size())
        # self.layer2.bias.data = fanin(self.layer2.bias.data.size())

    def forward(self, x):
        x = self.to_torch_variable(x)
        x = F.relu(self.layer1(x))
        self.layer1_w = self.layer1.weight.data.cpu().numpy()
        self.layer1_act = x.data.cpu().numpy()
        x = self.bn1(x)
        x = F.relu(self.layer2(x))
        self.layer2_w = self.layer2.weight.data.cpu().numpy()
        self.layer2_act = x.data.cpu().numpy()
        x = self.bn2(x)
        x = self.layer3(x)
        self.layer3_w = self.layer3.weight.data.cpu().numpy()
        self.layer3_act = x.data.cpu().numpy()
        x = self.action_scale * self.action_gate(x)
        return x

    def predict(self, x, to_numpy=True):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y

class DDPGCriticNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 gpu=False):
        super(DDPGCriticNet, self).__init__()
        hidden1 = 400
        hidden2 = 300
        self.layer1 = nn.Linear(state_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.layer2 = nn.Linear(hidden1 + action_dim, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.layer3 = nn.Linear(hidden2, 1)
        BasicNet.__init__(self, None, False, False)
        self.init_weights()

    def init_weights(self):
        bound = 3e-3
        self.layer3.weight.data.uniform_(-bound, bound)
        # self.layer3.bias.data.uniform_(-bound, bound)

        def fanin(size):
            v = 1.0 / np.sqrt(size[1])
            return torch.FloatTensor(size).uniform_(-v, v)

        self.layer1.weight.data = fanin(self.layer1.weight.data.size())
        # self.layer1.bias.data = fanin(self.layer1.bias.data.size())
        self.layer2.weight.data = fanin(self.layer2.weight.data.size())
        # self.layer2.bias.data = fanin(self.layer2.bias.data.size())

    def forward(self, x, action):
        x = self.to_torch_variable(x)
        action = self.to_torch_variable(action)
        x = F.relu(self.layer1(x))
        x = self.bn1(x)
        x = F.relu(self.layer2(torch.cat([x, action], dim=1)))
        x = self.bn2(x)
        x = self.layer3(x)
        return x

    def predict(self, x, action):
        return self.forward(x, action)

class GaussianActorNet(nn.Module, BasicNet):
    def __init__(self, state_dim, action_dim, action_scale=1.0, action_gate=None, gpu=False, unit_std=True):
        super(GaussianActorNet, self).__init__()
        hidden_size = 64
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

        BasicNet.__init__(self, None, gpu, False)

    def forward(self, x):
        x = self.to_torch_variable(x)
        phi = F.tanh(self.fc1(x))
        phi = F.tanh(self.fc2(phi))
        mean = self.action_mean(phi)
        if self.action_gate is not None:
            mean = self.action_scale * self.action_gate(mean)
        if self.unit_std:
            log_std = self.action_log_std.expand_as(mean)
            std = log_std.exp()
        else:
            std = F.softplus(self.fc_std(x) + 1e-5)
            log_std = std.log()
        return mean, std, log_std

    def predict(self, x):
        return self.forward(x)

    def log_density(self, x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(torch.FloatTensor([np.pi])).expand_as(x)) - log_std
        return log_density.sum(1)

    def entropy(self, std):
        return 0.5 * (1 + (2 * std.pow(2) * np.pi + 1e-5).log()).sum(1).mean()

class GaussianCriticNet(nn.Module, BasicNet):
    def __init__(self, state_dim, gpu=False):
        super(GaussianCriticNet, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        BasicNet.__init__(self, None, gpu, False)

    def forward(self, x):
        x = self.to_torch_variable(x)
        phi = F.tanh(self.fc1(x))
        phi = F.tanh(self.fc2(phi))
        value = self.fc_value(phi)
        return value

    def predict(self, x):
        return self.forward(x)

class DisjointActorCriticNet:
    def __init__(self, actor_network_fn, critic_network_fn):
        self.actor = actor_network_fn()
        self.critic = critic_network_fn()

    def state_dict(self):
        return [self.actor.state_dict(), self.critic.state_dict()]

    def load_state_dict(self, state_dicts):
        self.actor.load_state_dict(state_dicts[0])
        self.critic.load_state_dict(state_dicts[1])

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def zero_grad(self):
        self.actor.zero_grad()
        self.critic.zero_grad()
