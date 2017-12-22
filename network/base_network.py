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
        self.gpu = gpu and torch.cuda.is_available()
        self.LSTM = LSTM
        if self.gpu:
            self.cuda()
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

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
            if type(y) is list:
                y = [y_.cpu().data.numpy() for y_ in y]
            else:
                y = y.cpu().data.numpy()
        return y

# Base class for actor critic method
class ActorCriticNet(BasicNet):
    def predict(self, x):
        phi = self.forward(x, True)
        pre_prob = self.fc_actor(phi)
        prob = F.softmax(pre_prob, dim=1)
        log_prob = F.log_softmax(pre_prob, dim=1)
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