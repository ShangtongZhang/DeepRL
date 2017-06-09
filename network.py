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
            print 'Transferring network to GPU...'
            self.cuda()
            print 'Network transferred.'

    def to_torch_variable(self, x, dtype='float32'):
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
class FullyConnectedNet(nn.Module, VanillaNet):
    def __init__(self, dims, optimizer_fn=None, gpu=True):
        super(FullyConnectedNet, self).__init__()
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
class DuelingFullyConnectedNet(nn.Module, DuelingNet):
    def __init__(self, dims, optimizer_fn=None, gpu=True):
        super(DuelingFullyConnectedNet, self).__init__()
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

# Network for pixel Atari game with value based methods
class ConvNet(nn.Module, VanillaNet):
    def __init__(self, in_channels, n_actions, optimizer_fn=None, gpu=True):
        super(ConvNet, self).__init__()
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

class NipsConvNet(nn.Module, VanillaNet):
    def __init__(self, in_channels, n_actions, optimizer_fn=None, gpu=True):
        super(NipsConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc3 = nn.Linear(9 * 9 * 32, 256)
        self.fc4 = nn.Linear(256, n_actions)
        self.criterion = nn.MSELoss()
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc3(y))
        return self.fc4(y)

# Network for pixel Atari game with dueling architecture
class DuelingConvNet(nn.Module, DuelingNet):
    def __init__(self, in_channels, n_actions, optimizer_fn=None, gpu=True):
        super(DuelingConvNet, self).__init__()
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

# Network for CartPole with actor critic
class FCActorCriticNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 dims,
                 LSTM=False):
        super(FCActorCriticNet, self).__init__()
        if LSTM:
            self.layer1 = nn.LSTMCell(dims[0], dims[1])
        else:
            self.layer1 = nn.Linear(dims[0], dims[1])
        self.fc_actor = nn.Linear(dims[1], dims[2])
        self.fc_critic = nn.Linear(dims[1], 1)
        BasicNet.__init__(self, optimizer_fn=None, gpu=False, LSTM=LSTM)
        if LSTM:
            self.h = self.to_torch_variable(np.zeros((1, dims[1])))
            self.c = self.to_torch_variable(np.zeros((1, dims[1])))

    def forward(self, x, update_LSTM=True):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        if self.LSTM:
            h, c = self.layer1(x, (self.h, self.c))
            if update_LSTM:
                self.h = h
                self.c = c
            phi = h
        else:
            phi = self.layer1(x)
        return phi

# Network for pixel Atari game with actor critic
class ConvActorCriticNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 xentropy_weight=0.01,
                 grad_threshold=40,
                 gpu=True):
        super(ConvActorCriticNet, self).__init__()
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

class OpenAIConvActorCriticNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 LSTM=False):
        super(OpenAIConvActorCriticNet, self).__init__()
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
                 n_actions,
                 LSTM=False):
        super(OpenAIConvNet, self).__init__()
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

        self.fc6 = nn.Linear(hidden_units, n_actions)
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
        return self.fc6(phi)
