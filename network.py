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
    def __init__(self, optimizer_fn, gpu):
        if optimizer_fn is not None:
            self.optimizer = optimizer_fn(self.parameters())
        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            print 'Transferring network to GPU...'
            self.cuda()
            print 'Network transferred.'

    def to_torch_variable(self, x, dtype='float32'):
        x = torch.from_numpy(np.asarray(x, dtype=dtype))
        if self.gpu:
            x = x.cuda()
        return Variable(x)

# Base class for value based methods
class VanillaNet(BasicNet):
    def predict(self, x, to_numpy=True):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y

    def gradient(self, x, actions, targets):
        y = self.forward(x)
        y = y.gather(1, actions)
        loss = self.criterion(y, targets)
        loss.backward()

# Base class for actor critic method
class ActorCriticNet(BasicNet):
    def predict(self, x):
        phi = self.forward(x)
        return F.softmax(self.fc_actor(phi)).cpu().data.numpy()

    def gradient(self, x, actions, rewards):
        phi = self.forward(x)
        logit = self.fc_actor(phi)
        prob = F.softmax(logit)
        log_prob_ = F.log_softmax(logit)
        state_value = self.fc_critic(phi)
        log_prob = log_prob_.gather(1, actions)
        advantage = (rewards - state_value).detach()
        policy_loss = -torch.sum(log_prob * advantage)
        value_loss = 0.5 * torch.sum(torch.pow(rewards - state_value, 2))
        entropy = -torch.sum(torch.mul(prob, log_prob_))
        loss = policy_loss + value_loss - self.xentropy_weight * entropy
        loss.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.grad_threshold)

    def critic(self, x):
        phi = self.forward(x)
        return self.fc_critic(phi).cpu().data.numpy()

# Base class for dueling architecture
class DuelingNet(BasicNet):
    def predict(self, x, to_numpy=True):
        phi = self.forward(x)
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1).expand_as(advantange))
        if to_numpy:
            return q.cpu().data.numpy()
        return q

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
                 xentropy_weight=0.01,
                 grad_threshold=40,
                 gpu=True):
        super(FCActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc_actor = nn.Linear(dims[1], dims[2])
        self.fc_critic = nn.Linear(dims[1], 1)
        self.xentropy_weight = xentropy_weight
        self.grad_threshold = grad_threshold
        BasicNet.__init__(self, optimizer_fn=None, gpu=gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        phi = self.fc1(x)
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

