#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .base_network import *

class FCNet(nn.Module, VanillaNet):
    def __init__(self, state_dim, hidden_size, action_dim, gpu=-1):
        super(FCNet, self).__init__()
        self.fc_body = TwoLayerFCNet(state_dim, hidden_size)
        VanillaNet.__init__(self, hidden_size, action_dim, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.fc_body(x)

class DuelingFCNet(nn.Module, DuelingNet):
    def __init__(self, state_dim, hidden_size, action_dim, gpu=-1):
        super(DuelingFCNet, self).__init__()
        self.fc_body = TwoLayerFCNet(state_dim, hidden_size)
        DuelingNet.__init__(self, hidden_size, action_dim, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.fc_body(x)

class ActorCriticFCNet(nn.Module, ActorCriticNet):
    def __init__(self, state_dim, hidden_size, action_dim, gpu=-1):
        super(ActorCriticFCNet, self).__init__()
        self.fc_body = TwoLayerFCNet(state_dim, hidden_size)
        ActorCriticNet.__init__(self, hidden_size, action_dim, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.fc_body(x)

class CategoricalFCNet(nn.Module, CategoricalNet):
    def __init__(self, state_dim, n_actions, n_atoms, gpu=-1):
        super(CategoricalFCNet, self).__init__()
        hidden_size = 64
        self.fc_body = TwoLayerFCNet(state_dim, hidden_size)
        CategoricalNet.__init__(self, hidden_size, n_actions, n_atoms, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.fc_body(x)

class QuantileFCNet(nn.Module, QuantileNet):
    def __init__(self, state_dim, n_actions, n_quantiles, gpu=-1):
        super(QuantileFCNet, self).__init__()
        hidden_size = 64
        self.fc_body = TwoLayerFCNet(state_dim, hidden_size)
        QuantileNet.__init__(self, hidden_size, n_actions, n_quantiles, gpu)

    def feature(self, x):
        x = self.variable(x)
        return self.fc_body(x)
