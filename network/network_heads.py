#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body, gpu=-1):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class ActorCriticNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(ActorCriticNet, self).__init__()
        self.fc_actor = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.fc_critic = layer_init(nn.Linear(body.feature_dim, 1))
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        pre_prob = self.fc_actor(phi)
        prob = F.softmax(pre_prob, dim=1)
        log_prob = F.log_softmax(pre_prob, dim=1)
        value = self.fc_critic(phi)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob, log_prob, value

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body, gpu=-1):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body, gpu=-1):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.cpu().detach().numpy()
        return quantiles

class GaussianActorNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(GaussianActorNet, self).__init__()
        self.fc_action = layer_init(nn.Linear(body.feature_dim, action_dim), 3e-3)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x):
        x = self.tensor(x)
        phi = self.body(x)
        mean = F.tanh(self.fc_action(phi))
        log_std = self.action_log_std.expand_as(mean)
        std = log_std.exp()
        return mean, std, log_std

class GaussianCriticNet(nn.Module, BaseNet):
    def __init__(self, body, gpu=-1):
        super(GaussianCriticNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1), 3e-3)
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x):
        x = self.tensor(x)
        phi = self.body(x)
        value = self.fc_value(phi)
        return value

class DeterministicActorNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(DeterministicActorNet, self).__init__()
        self.fc_action = layer_init(nn.Linear(body.feature_dim, action_dim), 3e-3)
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        x = self.tensor(x)
        phi = self.body(x)
        a = F.tanh(self.fc_action(phi))
        if to_numpy:
            a = a.cpu().detach().numpy()
        return a

class DeterministicCriticNet(nn.Module, BaseNet):
    def __init__(self, body, gpu=-1):
        super(DeterministicCriticNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1), 3e-3)
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, action):
        x = self.tensor(x)
        action = self.tensor(action)
        phi = self.body(x, action)
        value = self.fc_value(phi)
        return value

class EnvModel(nn.Module):
    def __init__(self, phi_dim, action_dim):
        super(EnvModel, self).__init__()
        self.hidden_dim = 300
        self.fc_r1 = layer_init(nn.Linear(phi_dim + action_dim, self.hidden_dim))
        self.fc_r2 = layer_init(nn.Linear(self.hidden_dim, 1))

        self.fc_t1 = layer_init(nn.Linear(phi_dim, phi_dim))
        self.fc_t2 = layer_init(nn.Linear(phi_dim + action_dim, phi_dim))

    def forward(self, phi_s, action):
        phi = torch.cat([phi_s, action], dim=1)
        r = self.fc_r2(F.tanh(self.fc_r1(phi)))

        phi_s_prime = phi_s + F.tanh(self.fc_t1(phi_s))
        phi_sa_prime = torch.cat([phi_s_prime, action], dim=1)
        phi_s_prime = phi_s_prime + F.tanh(self.fc_t2(phi_sa_prime))

        return r, phi_s_prime

from .network_bodies import *
class SharedDeterministicNet(nn.Module, BaseNet):
    def __init__(self, state_dim, action_dim, discount, detach_action=False, gate=F.tanh, num_models=1, gpu=-1):
        super(SharedDeterministicNet, self).__init__()
        self.phi_dim = 400
        self.hidden_dim = 300

        self.fc_phi = layer_init(nn.Linear(state_dim, self.phi_dim))

        self.fc_q1 = layer_init(nn.Linear(self.phi_dim + action_dim, self.hidden_dim))
        self.fc_q2 = layer_init(nn.Linear(self.hidden_dim, 1), 3e-3)

        self.fc_a1 = layer_init(nn.Linear(self.phi_dim, self.hidden_dim))
        self.fc_a2 = layer_init(nn.Linear(self.hidden_dim, action_dim), 3e-3)

        self.models = nn.ModuleList([EnvModel(self.phi_dim, action_dim) for _ in range(num_models)])

        self.gate = gate
        self.discount = discount
        self.detach_action = detach_action

        self.set_gpu(gpu)

    def env_model(self, phi_s, action):
        phi_s_primes, rs = zip(*[m(phi_s, action) for m in self.models])
        phi_s_primes = torch.stack(phi_s_primes, 0)
        rs = torch.stack(rs, 0)
        return phi_s_primes, rs

    # def compute_r(self, phi_s, action):
    #     phi = torch.cat([phi_s, action], dim=1)
    #     r = self.fc_r2(F.tanh(self.fc_r1(phi)))
    #     return r

    def comupte_q(self, phi_s, action):
        phi = torch.cat([phi_s, action], dim=-1)
        q = self.fc_q2(F.tanh(self.fc_q1(phi)))
        return q

    def compute_a(self, phi_s):
        return F.tanh(self.fc_a2(F.tanh(self.fc_a1(phi_s))))

    def compute_phi(self, obs):
        return F.tanh(self.fc_phi(obs))

    # def compute_phi_prime(self, phi_s, action):
    #     phi_s_prime = phi_s + F.tanh(self.fc_t1(phi_s))
    #     phi_sa_prime = torch.cat([phi_s_prime, action], dim=1)
    #     phi_s_prime = phi_s_prime + F.tanh(self.fc_t2(phi_sa_prime))
    #     return phi_s_prime

    def actor(self, obs):
        obs = self.tensor(obs)
        phi_s = self.compute_phi(obs)
        return self.compute_a(phi_s)

    def critic(self, obs, a, lam=0):
        obs = self.tensor(obs)
        a = self.tensor(a)
        phi_s = self.compute_phi(obs)
        q0 = self.comupte_q(phi_s, a)

        phi_s_primes, rs = self.env_model(phi_s, a)
        a_primes = self.compute_a(phi_s_primes)
        q_primes = self.comupte_q(phi_s_primes, a_primes)
        q1 = rs + self.discount * q_primes
        q1 = q1
        # a_primes = [self.compute_a(phi_s_prime) for phi_s_prime in phi_s_primes]

        r = self.compute_r(phi_s, a)

        phi_s_prime = self.compute_phi_prime(phi_s, a)
        a_prime = self.compute_a(phi_s_prime)
        if self.detach_action:
            a_prime = a_prime.detach()
        q_prime = self.comupte_q(phi_s_prime, a_prime)
        q1 = r + self.discount * q_prime

        q = lam * q0 + (1 - lam) * q1

        return q, r
