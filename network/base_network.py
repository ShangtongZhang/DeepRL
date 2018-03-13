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
    def __init__(self, gpu, LSTM=False):
        if not torch.cuda.is_available():
            gpu = -1
        self.gpu = gpu
        self.LSTM = LSTM
        self.init_weights()
        if self.gpu >= 0:
            self.cuda(self.gpu)

    def supported_dtype(self, x, torch_type):
        if torch_type == torch.FloatTensor:
            return np.asarray(x, dtype=np.float32)
        if torch_type == torch.LongTensor:
            return np.asarray(x, dtype=np.int64)

    def variable(self, x, dtype=torch.FloatTensor):
        if isinstance(x, Variable):
            return x
        x = dtype(torch.from_numpy(self.supported_dtype(x, dtype)))
        if self.gpu >= 0:
            x = x.cuda(self.gpu)
        return Variable(x)

    def tensor(self, x, dtype=torch.FloatTensor):
        x = dtype(torch.from_numpy(self.supported_dtype(x, dtype)))
        if self.gpu >= 0:
            x = x.cuda(self.gpu)
        return x

    def reset(self, terminal):
        if not self.LSTM:
            return
        if terminal:
            self.h.data.zero_()
            self.c.data.zero_()
        self.h = Variable(self.h.data)
        self.c = Variable(self.c.data)

    def init_weights(self):
        for layer in self.children():
            relu_gain = nn.init.calculate_gain('relu')
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal(layer.weight.data, relu_gain)
            nn.init.constant(layer.bias.data, 0)

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

class CategoricalNet(BasicNet):
    def predict(self, x, to_numpy=False):
        phi = self.forward(x)
        pre_prob = self.fc_categorical(phi).view((-1, self.n_actions, self.n_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().data.numpy()
        return prob
