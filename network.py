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
from SMDWrapper import SMDWrapper

class FullyConnectedNet(nn.Module):
    def __init__(self, dims, learning_rate, gpu=True):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.parameters(), learning_rate)
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

    def output_transfer(self, y):
        return y

class SMDNetworkWrapper(SMDWrapper):
    def __init__(self, net):
        SMDWrapper.__init__(self, net)

    def sync_with(self, src_net):
        self.net.sync_with(src_net.net)

    def parameters(self):
        return self.net.parameters()
