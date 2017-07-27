#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *

class ContinuousActorCriticNet(nn.Module, BasicNet):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ContinuousActorCriticNet, self).__init__()
        hidden_size1 = 64
        hidden_size2 = 64
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc_mean = nn.Linear(hidden_size2, action_dim)
        self.fc_var = nn.Linear(hidden_size2, action_dim)
        self.fc_critic = nn.Linear(hidden_size2, 1)
        BasicNet.__init__(self, None, False)

    def forward(self, x):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        phi = F.relu(self.fc2(x))
        return phi

    def predict(self, x):
        phi = self.forward(x)
        mean = self.fc_mean(phi)
        var = F.softplus(self.fc_var(phi) + 1e-5)
        value = self.fc_critic(phi)
        return mean, var, value

    def critic(self, x):
        phi = self.forward(x)
        return self.fc_critic(phi)

class DDPGActorNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 output_gate,
                 gpu=False):
        super(DDPGActorNet, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.output_gate = output_gate
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
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # x = self.output_gate(self.layer3(x))
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
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400 + action_dim, 300)
        self.layer3 = nn.Linear(300, 1)
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
        x = F.relu(self.layer2(torch.cat([x, action], dim=1)))
        x = self.layer3(x)
        return x

    def predict(self, x, action):
        return self.forward(x, action)