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
        x = torch.from_numpy(np.asarray(x, dtype='float32'))
        if self.gpu:
            x = x.cuda()
        x = Variable(x)

        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

    def sync_with(self, src_net):
        for param_dst, param_src in zip(self.parameters(), src_net.parameters()):
            param_dst.data.copy_(param_src.data)

    def predict(self, x):
        return self.forward(x).cpu().data.numpy()

    def learn(self, x, target):
        target = torch.from_numpy(target)
        if self.gpu:
            target = target.cuda()
        target = Variable(target)
        y = self.forward(x)
        loss = self.criterion(y, target)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()

    def gradient(self, x, actions, rewards):
        y = self.forward(x)
        target = np.copy(y.data.numpy())
        target[np.arange(target.shape[0]), actions] = np.asarray(rewards)
        target = Variable(torch.from_numpy(target))
        loss = self.criterion(y, target)
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

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype='float32'))
        if self.gpu:
            x = x.cuda()
        x = Variable(x)
        phi = self.fc1(x)
        return phi

    def predict(self, x):
        phi = self.forward(x)
        return F.softmax(self.fc_actor(phi)).cpu().data.numpy()

    def gradient(self, x, actions, rewards):
        phi = self.forward(x)
        logit = self.fc_actor(phi)
        log_prob = F.log_softmax(logit)
        state_value = self.fc_critic(phi)
        log_prob = log_prob.gather(1, Variable(torch.from_numpy(np.asarray([actions]).reshape([-1, 1]))))
        advantage = np.asarray([rewards]).reshape([-1, 1]) - state_value.cpu().data.numpy()
        policy_loss = -torch.sum(log_prob * Variable(torch.from_numpy(np.asarray(advantage, dtype='float32'))))
        value_loss = 0.5 * torch.sum(torch.pow(state_value - Variable(torch.from_numpy(np.asarray(rewards, dtype='float32'))), 2))
        (policy_loss + value_loss).backward()
        nn.utils.clip_grad_norm(self.parameters(), 40)

    def critic(self, x):
        phi = self.forward(x)
        return self.fc_critic(phi).cpu().data.numpy()