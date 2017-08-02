#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *

class ContinuousActorCriticNet(nn.Module, BasicNet):
    def __init__(self, state_dim, action_dim, action_scale, action_gate):
        super(ContinuousActorCriticNet, self).__init__()
        actor_hidden = 200
        critic_hidden = 100
        self.fc_actor = nn.Linear(state_dim, actor_hidden)
        self.fc_mean = nn.Linear(actor_hidden, action_dim)
        self.fc_std = nn.Linear(actor_hidden, action_dim)
        self.action_scale = action_scale
        self.action_gate = action_gate
        self.actor_params = list(self.fc_actor.parameters()) + \
                            list(self.fc_mean.parameters()) + \
                            list(self.fc_std.parameters())

        self.fc_critic = nn.Linear(state_dim, critic_hidden)
        self.fc_value = nn.Linear(critic_hidden, 1)
        self.critic_params = list(self.fc_critic.parameters()) + \
                             list(self.fc_value.parameters())

        BasicNet.__init__(self, None, False)

    def predict(self, x):
        x = self.to_torch_variable(x)
        value = self.critic(x)

        x = F.relu(self.fc_actor(x))
        mean = self.action_scale * self.action_gate(self.fc_mean(x))
        std = F.softplus(self.fc_std(x) + 1e-5)

        return mean, std, value

    def critic(self, x):
        x = self.to_torch_variable(x)
        x = F.relu(self.fc_critic(x))
        x = self.fc_value(x)
        return x

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