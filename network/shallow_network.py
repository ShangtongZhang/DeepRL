#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .base_network import *

# Network for CartPole with value based methods
class FCNet(nn.Module, VanillaNet):
    def __init__(self, dims, gpu=0):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

# Network for CartPole with dueling architecture
class DuelingFCNet(nn.Module, DuelingNet):
    def __init__(self, dims, gpu=0):
        super(DuelingFCNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc_value = nn.Linear(dims[2], 1)
        self.fc_advantage = nn.Linear(dims[2], dims[3])
        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        y = F.relu(self.fc1(x))
        phi = F.relu(self.fc2(y))
        return phi

# Network for CartPole with actor critic
class ActorCriticFCNet(nn.Module, ActorCriticNet):
    def __init__(self, state_dim, action_dim, gpu=-1):
        super(ActorCriticFCNet, self).__init__()
        hidden_size1 = 64
        hidden_size2 = 64
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc_actor = nn.Linear(hidden_size2, action_dim)
        self.fc_critic = nn.Linear(hidden_size2, 1)
        BasicNet.__init__(self, gpu=gpu)

    def forward(self, x, update_LSTM=True):
        x = self.variable(x)
        x = F.relu(self.fc1(x))
        phi = F.relu(self.fc2(x))
        return phi

class CategoricalFCNet(nn.Module, CategoricalNet):
    def __init__(self, state_dim, n_actions, n_atoms, gpu=0):
        super(CategoricalFCNet, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        hidden_size = 64
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_categorical = nn.Linear(hidden_size, n_actions * n_atoms)
        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        phi = F.relu(self.fc1(x))
        phi = F.relu(self.fc2(phi))
        return phi

class QuantileFCNet(nn.Module, QuantileNet):
    def __init__(self, state_dim, n_actions, n_quantiles, gpu=0):
        super(QuantileFCNet, self).__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        hidden_size = 64
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions * n_quantiles)
        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        phi = F.relu(self.fc1(x))
        phi = F.relu(self.fc2(phi))
        quantiles = self.fc3(phi)
        return quantiles

class GammaFCNet(nn.Module, GammaNet):
    def __init__(self, state_dim, action_dim, num_peers, gpu=-1):
        super(GammaFCNet, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_actor = nn.Linear(hidden_size * num_peers, action_dim)
        self.fc_critic = nn.Linear(hidden_size * num_peers, 1)

        self.fc_attention = nn.Linear(hidden_size, num_peers - 1)
        self.fc_q = nn.Linear(hidden_size, action_dim)

        # self.fc_actor_main = nn.Linear(hidden_size, action_dim)
        # self.fc_critic_main = nn.Linear(hidden_size, 1)
        self.compute_attention = self.softmax_attention
        BasicNet.__init__(self, gpu=gpu)

    def forward(self, x, update_lstm=True):
        x = self.variable(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class GammaAttentionFCNet(nn.Module, GammaAttentionNet):
    def __init__(self, state_dim, action_dim, num_peers, gpu=-1):
        super(GammaAttentionFCNet, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_actor = nn.Linear(hidden_size, action_dim)
        self.fc_critic = nn.Linear(hidden_size, 1)

        self.fc_attention = nn.Linear(hidden_size, num_peers - 1)
        self.fc_q = nn.Linear(hidden_size, action_dim)
        BasicNet.__init__(self, gpu=gpu)
        self.fc_attention.weight.data.zero_()

    def forward(self, x, update_lstm=True):
        x = self.variable(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
