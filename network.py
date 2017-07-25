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

# Base class for all kinds of network
class BasicNet:
    def __init__(self, optimizer_fn, gpu, LSTM=False):
        if optimizer_fn is not None:
            self.optimizer = optimizer_fn(self.parameters())
        self.gpu = gpu and torch.cuda.is_available()
        self.LSTM = LSTM
        if self.gpu:
            self.cuda()

    def to_torch_variable(self, x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        if self.gpu:
            x = x.cuda()
        return Variable(x)

    def reset(self, terminal):
        if not self.LSTM:
            return
        if terminal:
            self.h.data.zero_()
            self.c.data.zero_()
        self.h = Variable(self.h.data)
        self.c = Variable(self.c.data)

# Base class for value based methods
class VanillaNet(BasicNet):
    def predict(self, x, to_numpy=False):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y

# Base class for actor critic method
class ActorCriticNet(BasicNet):
    def predict(self, x):
        phi = self.forward(x, True)
        pre_prob = self.fc_actor(phi)
        prob = F.softmax(pre_prob)
        log_prob = F.log_softmax(pre_prob)
        value = self.fc_critic(phi)
        return prob, log_prob, value

    def critic(self, x):
        phi = self.forward(x, False)
        return self.fc_critic(phi)

# Base class for dueling architecture
class DuelingNet(BasicNet):
    def predict(self, x, to_numpy=False):
        phi = self.forward(x)
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1).expand_as(advantange))
        if to_numpy:
            return q.cpu().data.numpy()
        return q

# Starting of several network instances

# Network for CartPole with value based methods
class FCNet(nn.Module, VanillaNet):
    def __init__(self, dims, optimizer_fn=None, gpu=True):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.criterion = nn.MSELoss()
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
        self.criterion = nn.MSELoss()
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        y = F.relu(self.fc1(x))
        phi = F.relu(self.fc2(y))
        return phi

    # Network for CartPole with actor critic
class ActorCriticFCNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 dims):
        super(ActorCriticFCNet, self).__init__()
        self.layer1 = nn.Linear(dims[0], dims[1])
        self.fc_actor = nn.Linear(dims[1], dims[2])
        self.fc_critic = nn.Linear(dims[1], 1)
        BasicNet.__init__(self, None, False)

    def forward(self, x, update_LSTM=True):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        phi = self.layer1(x)
        return phi

# Network for pixel Atari game with value based methods
class NatureConvNet(nn.Module, VanillaNet):
    def __init__(self, in_channels, n_actions, optimizer_fn=None, gpu=True):
        super(NatureConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)
        self.criterion = nn.MSELoss()
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return self.fc5(y)

# Network for pixel Atari game with dueling architecture
class DuelingNatureConvNet(nn.Module, DuelingNet):
    def __init__(self, in_channels, n_actions, optimizer_fn=None, gpu=True):
        super(DuelingNatureConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_advantage = nn.Linear(512, n_actions)
        self.fc_value = nn.Linear(512, 1)
        self.criterion = nn.MSELoss()
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        phi = F.relu(self.fc4(y))
        return phi



# Network for pixel Atari game with actor critic
class ActorCriticNatureConvNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 xentropy_weight=0.01,
                 grad_threshold=40,
                 gpu=True):
        super(ActorCriticNatureConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_actor = nn.Linear(512, n_actions)
        self.fc_critic = nn.Linear(512, 1)
        self.xentropy_weight = xentropy_weight
        self.grad_threshold = grad_threshold
        BasicNet.__init__(self, optimizer_fn=None, gpu=gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = y.view(y.size(0), -1)
        return F.elu(self.fc4(y))

class OpenAIActorCriticConvNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 LSTM=False):
        super(OpenAIActorCriticConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.LSTM = LSTM
        hidden_units = 256

        if LSTM:
            self.layer5 = nn.LSTMCell(32 * 3 * 3, hidden_units)
        else:
            self.layer5 = nn.Linear(32 * 3 * 3, hidden_units)

        self.fc_actor = nn.Linear(hidden_units, n_actions)
        self.fc_critic = nn.Linear(hidden_units, 1)
        BasicNet.__init__(self, optimizer_fn=None, gpu=False, LSTM=LSTM)
        if LSTM:
            self.h = self.to_torch_variable(np.zeros((1, hidden_units)))
            self.c = self.to_torch_variable(np.zeros((1, hidden_units)))

    def forward(self, x, update_LSTM=True):
        x = self.to_torch_variable(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.view(y.size(0), -1)
        if self.LSTM:
            h, c = self.layer5(y, (self.h, self.c))
            if update_LSTM:
                self.h = h
                self.c = c
            phi = h
        else:
            phi = F.elu(self.layer5(y))
        return phi

class OpenAIConvNet(nn.Module, VanillaNet):
    def __init__(self,
                 in_channels,
                 n_actions):
        super(OpenAIConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        hidden_units = 256
        self.layer5 = nn.Linear(32 * 3 * 3, hidden_units)
        self.fc6 = nn.Linear(hidden_units, n_actions)

        BasicNet.__init__(self, optimizer_fn=None, gpu=False, LSTM=False)

    def forward(self, x, update_LSTM=True):
        x = self.to_torch_variable(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.view(y.size(0), -1)
        phi = F.elu(self.layer5(y))
        return self.fc6(phi)

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
