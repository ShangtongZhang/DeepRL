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


class FullyConnectedNet(nn.Module):
    def __init__(self, dims, optimizer_fn=None, gpu=True):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.criterion = nn.MSELoss()
        if optimizer_fn is not None:
            self.optimizer = optimizer_fn(self.parameters())
        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            print 'Transferring network to GPU...'
            self.cuda()
            print 'Network transferred.'

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        x = self.to_torch_variable(x)

        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

    def predict(self, x):
        return self.forward(x).cpu().data.numpy()

    def to_torch_variable(self, x, dtype='float32'):
        x = torch.from_numpy(np.asarray(x, dtype=dtype))
        if self.gpu:
            x = x.cuda()
        return Variable(x)

    def learn(self, x, actions, targets):
        self.zero_grad()
        self.gradient(x, actions, targets)
        self.optimizer.step()

    # def clippedLearn(self, x, actions, targets):
    #     y = self.forward(x)
    #     actions = self.to_torch_variable(actions, 'int64').unsqueeze(1)
    #     targets = self.to_torch_variable(targets).unsqueeze(1)
    #     y = y.gather(1, actions)
    #     bellman_error = targets - y
    #     bellman_error = bellman_error.clamp(-1, 1) * -1
    #     self.zero_grad()
    #     y.backward(bellman_error.data)
    #     self.optimizer.step()

    def gradient(self, x, actions, targets):
        y = self.forward(x)
        actions = self.to_torch_variable(actions, 'int64').unsqueeze(1)
        targets = self.to_torch_variable(targets).unsqueeze(1)
        y = y.gather(1, actions)
        loss = self.criterion(y, targets)
        loss.backward()

    def output_transfer(self, y):
        return y


class ActorCriticNet(nn.Module):
    def __init__(self, dims, gpu=True):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc_actor = nn.Linear(dims[1], dims[2])
        self.fc_critic = nn.Linear(dims[1], 1)
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

    def forward(self, x):
        phi = self.fc1(self.to_torch_variable(x))
        return phi

    def predict(self, x):
        phi = self.forward(x)
        return F.softmax(self.fc_actor(phi)).cpu().data.numpy()

    def gradient(self, x, actions, rewards):
        phi = self.forward(x)
        logit = self.fc_actor(phi)
        prob = F.softmax(logit)
        log_prob_ = F.log_softmax(logit)
        state_value = self.fc_critic(phi)
        log_prob = log_prob_.gather(1, self.to_torch_variable(np.asarray([actions]).reshape([-1, 1]), 'int64'))
        advantage = np.asarray([rewards]).reshape([-1, 1]) - state_value.cpu().data.numpy()
        policy_loss = -torch.sum(log_prob * self.to_torch_variable(advantage))
        value_loss = 0.5 * torch.sum(
            torch.pow(state_value - Variable(torch.from_numpy(np.asarray(rewards, dtype='float32'))), 2))
        entropy = -torch.sum(torch.mul(prob, log_prob_))
        (policy_loss + value_loss - 0.01 * entropy).backward()
        nn.utils.clip_grad_norm(self.parameters(), 40)

    def critic(self, x):
        phi = self.forward(x)
        return self.fc_critic(phi).cpu().data.numpy()


class ConvNet(nn.Module):
    def __init__(self, in_channels, n_actions, optimizer_fn=None, gpu=True):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)

        self.criterion = nn.MSELoss()
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

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return self.fc5(y)

    def predict(self, x):
        return self.forward(self.to_torch_variable(x)).cpu().data.numpy()

    def learn(self, x, actions, targets):
        self.zero_grad()
        self.gradient(x, actions, targets)
        self.optimizer.step()

    def gradient(self, x, actions, targets):
        y = self.forward(self.to_torch_variable(x))
        actions = self.to_torch_variable(actions, 'int64').unsqueeze(1)
        targets = self.to_torch_variable(targets).unsqueeze(1)
        y = y.gather(1, actions)
        loss = self.criterion(y, targets)
        loss.backward()
