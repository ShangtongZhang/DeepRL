#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .base_network import *

class ConvNet(nn.Module, VanillaNet):
    def __init__(self, in_channels, action_dim, gpu=-1):
        super(ConvNet, self).__init__()
        self.body = NatureConvNet(in_channels)
        VanillaNet.__init__(self, self.body.feature_dim, action_dim, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.body(x)

class DuelingConvNet(nn.Module, DuelingNet):
    def __init__(self, in_channels, action_dim, gpu=-1):
        super(DuelingConvNet, self).__init__()
        self.body = NatureConvNet(in_channels)
        DuelingNet.__init__(self, self.body.feature_dim, action_dim, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.body(x)

class ActorCriticConvNet(nn.Module, ActorCriticNet):
    def __init__(self, in_channels, action_dim, gpu=-1):
        super(ActorCriticConvNet, self).__init__()
        self.body = NatureConvNet(in_channels)
        ActorCriticNet.__init__(self, self.body.feature_dim, action_dim, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.body(x)

class CategoricalConvNet(nn.Module, CategoricalNet):
    def __init__(self, in_channels, n_actions, n_atoms, gpu=-1):
        super(CategoricalConvNet, self).__init__()
        self.body = NatureConvNet(in_channels)
        CategoricalNet.__init__(self, self.body.feature_dim, n_actions, n_atoms, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.body(x)

class QuantileConvNet(nn.Module, QuantileNet):
    def __init__(self, in_channels, n_actions, n_quantiles, gpu=-1):
        super(QuantileConvNet, self).__init__()
        self.body = NatureConvNet(in_channels)
        QuantileNet.__init__(self, self.body.feature_dim, n_actions, n_quantiles, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.body(x)
