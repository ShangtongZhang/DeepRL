#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .base_network import *

# Network for CartPole with value based methods
class FCNet(nn.Module, VanillaNet):
    def __init__(self, dims, optimizer_fn=None, gpu=True):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

# Network for CartPole with dueling architecture
class DuelingFCNet(nn.Module, DuelingNet):
    def __init__(self, dims, optimizer_fn=None, gpu=True):
        super(DuelingFCNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc_value = nn.Linear(dims[2], 1)
        self.fc_advantage = nn.Linear(dims[2], dims[3])
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        y = F.relu(self.fc1(x))
        phi = F.relu(self.fc2(y))
        return phi

# Network for CartPole with actor critic
class ActorCriticFCNet(nn.Module, ActorCriticNet):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticFCNet, self).__init__()
        hidden_size1 = 50
        hidden_size2 = 200
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc_actor = nn.Linear(hidden_size2, action_dim)
        self.fc_critic = nn.Linear(hidden_size2, 1)
        BasicNet.__init__(self, None, False)

    def forward(self, x, update_LSTM=True):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        phi = self.fc2(x)
        return phi

class FruitHRFCNet(nn.Module, VanillaNet):
    def __init__(self, state_dim, action_dim, head_weights, optimizer_fn=None, gpu=True):
        super(FruitHRFCNet, self).__init__()
        hidden_size = 250
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.ModuleList([nn.Linear(hidden_size, action_dim) for _ in head_weights])
        self.head_weights = head_weights
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x, heads_only):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        head_q = [fc(x) for fc in self.fc2]
        if not heads_only:
            q = [h * w for h, w in zip(head_q, self.head_weights)]
            q = torch.stack(q, dim=0)
            q = q.sum(0).squeeze(0)
            return q
        else:
            return head_q

    def predict(self, x, heads_only):
        return self.forward(x, heads_only)

class FruitMultiStatesFCNet(nn.Module, BasicNet):
    def __init__(self, state_dim, action_dim, head_weights, optimizer_fn=None, gpu=True):
        super(FruitMultiStatesFCNet, self).__init__()
        hidden_size = 250
        self.fc1 = nn.ModuleList([nn.Linear(state_dim, hidden_size) for _ in head_weights])
        self.fc2 = nn.ModuleList([nn.Linear(hidden_size, action_dim) for _ in head_weights])
        self.head_weights = head_weights
        self.state_dim = state_dim
        self.n_heads = head_weights.shape[0]
        BasicNet.__init__(self, optimizer_fn, gpu)

    def predict(self, x, merge):
        head_q = []
        for i in range(self.n_heads):
            q = self.to_torch_variable(x[:, i, :])
            q = self.fc1[i](q)
            q = F.relu(q)
            q = self.fc2[i](q)
            head_q.append(q)
        if merge:
            q = [q * w for q, w in zip(head_q, self.head_weights)]
            q = torch.stack(q, dim=0)
            q = q.sum(0).squeeze(0)
            return q
        return head_q
