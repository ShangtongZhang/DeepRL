#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class TwoLayerFCBody(nn.Module):
    def __init__(self, state_dim, hidden_size=64, gate=F.relu):
        super(TwoLayerFCBody, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.gate = gate
        self.feature_dim = hidden_size

    def forward(self, x):
        y = self.gate(self.fc1(x))
        y = self.gate(self.fc2(y))
        return y

class DeterministicActorNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_gate=F.tanh,
                 action_scale=1,
                 gpu=-1,
                 non_linear=F.tanh):
        super(DeterministicActorNet, self).__init__()
        self.layer1 = layer_init(nn.Linear(state_dim, 300))
        self.layer2 = layer_init(nn.Linear(300, 200))
        self.layer3 = nn.Linear(200, action_dim)
        self.action_gate = action_gate
        self.action_scale = action_scale
        self.non_linear = non_linear
        self.init_weights()
        BasicNet.__init__(self, gpu)

    def init_weights(self):
        bound = 3e-3
        nn.init.uniform(self.layer3.weight.data, -bound, bound)
        nn.init.constant(self.layer3.bias.data, 0)

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
                 non_linear=F.tanh):
        super(DeterministicCriticNet, self).__init__()
        self.layer1 = layer_init(nn.Linear(state_dim, 400))
        self.layer2 = layer_init(nn.Linear(400 + action_dim, 300))
        self.layer3 = nn.Linear(300, 1)
        self.non_linear = non_linear
        self.init_weights()
        BasicNet.__init__(self, gpu)

    def init_weights(self):
        bound = 3e-3
        nn.init.uniform(self.layer3.weight.data, -bound, bound)
        nn.init.constant(self.layer3.bias.data, 0)

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
                 gpu=-1,
                 hidden_size=64,
                 non_linear=F.tanh):
        super(GaussianActorNet, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc_action = nn.Linear(hidden_size, action_dim)

        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.non_linear = non_linear

        self.init_weights()
        BasicNet.__init__(self, gpu)

    def init_weights(self):
        bound = 3e-3
        nn.init.uniform(self.fc_action.weight.data, -bound, bound)
        nn.init.constant(self.fc_action.bias.data, 0)

    def forward(self, x):
        x = self.variable(x)
        phi = self.non_linear(self.fc1(x))
        phi = self.non_linear(self.fc2(phi))
        mean = F.tanh(self.fc_action(phi))
        log_std = self.action_log_std.expand_as(mean)
        std = log_std.exp()
        return mean, std, log_std

    def predict(self, x):
        return self.forward(x)

class GaussianCriticNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 gpu=-1,
                 hidden_size=64,
                 non_linear=F.tanh):
        super(GaussianCriticNet, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc_value = nn.Linear(hidden_size, 1)
        self.non_linear = non_linear
        self.init_weights()
        BasicNet.__init__(self, gpu)

    def init_weights(self):
        bound = 3e-3
        nn.init.uniform(self.fc_value.weight.data, -bound, bound)
        nn.init.constant(self.fc_value.bias.data, 0)

    def forward(self, x):
        x = self.variable(x)
        phi = self.non_linear(self.fc1(x))
        phi = self.non_linear(self.fc2(phi))
        value = self.fc_value(phi)
        return value

    def predict(self, x):
        return self.forward(x)