#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class VanillaNet(nn.Module, BasicNet):
    def __init__(self, output_dim, body, gpu=-1):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.variable(x))
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y

class DuelingNet(nn.Module, BasicNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.variable(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().data.numpy()
        return q

class ActorCriticNet(nn.Module, BasicNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(ActorCriticNet, self).__init__()
        self.fc_actor = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.fc_critic = layer_init(nn.Linear(body.feature_dim, 1))
        self.body = body
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.variable(x))
        pre_prob = self.fc_actor(phi)
        prob = F.softmax(pre_prob, dim=1)
        log_prob = F.log_softmax(pre_prob, dim=1)
        value = self.fc_critic(phi)
        if to_numpy:
            return prob.cpu().data.numpy()
        return prob, log_prob, value

class CategoricalNet(nn.Module, BasicNet):
    def __init__(self, action_dim, num_atoms, body, gpu=-1):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.variable(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().data.numpy()
        return prob

class QuantileNet(nn.Module, BasicNet):
    def __init__(self, action_dim, num_quantiles, body, gpu=-1):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        BasicNet.__init__(self, gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.variable(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.data.cpu().numpy()
        return quantiles




