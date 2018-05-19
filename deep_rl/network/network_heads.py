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

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options, gpu=-1):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x):
        phi = self.body(self.tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

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
        phi = torch.cat([phi_s, action], dim=-1)
        r = self.fc_r2(F.tanh(self.fc_r1(phi)))

        phi_s_prime = phi_s + F.tanh(self.fc_t1(phi_s))
        phi_sa_prime = torch.cat([phi_s_prime, action], dim=-1)
        phi_s_prime = phi_s_prime + F.tanh(self.fc_t2(phi_sa_prime))

        return phi_s_prime, r

class ActorModel(nn.Module):
    def __init__(self, phi_dim, action_dim):
        super(ActorModel, self).__init__()
        self.hidden_dim = 300
        self.layers = nn.Sequential(
            layer_init(nn.Linear(phi_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, action_dim), 3e-3),
            nn.Tanh()
        )

    def forward(self, phi_s):
        return self.layers(phi_s)

class CriticModel(nn.Module):
    def __init__(self, phi_dim, action_dim):
        super(CriticModel, self).__init__()
        self.hidden_dim = 300
        self.layers = nn.Sequential(
            layer_init(nn.Linear(phi_dim + action_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, 1), 3e-3)
        )

    def forward(self, phi_s, action):
        phi = torch.cat([phi_s, action], dim=-1)
        return self.layers(phi)

class FeatureModel(nn.Module):
    def __init__(self, state_dim, phi_dim):
        super(FeatureModel, self).__init__()
        self.fc = layer_init(nn.Linear(state_dim, phi_dim))

    def forward(self, x):
        return F.tanh(self.fc(x))

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

    def comupte_q(self, phi_s, action):
        phi = torch.cat([phi_s, action], dim=-1)
        q = self.fc_q2(F.tanh(self.fc_q1(phi)))
        return q

    def compute_a(self, phi_s):
        return F.tanh(self.fc_a2(F.tanh(self.fc_a1(phi_s))))

    def compute_phi(self, obs):
        return F.tanh(self.fc_phi(obs))

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
        if self.detach_action:
            a_primes = a_primes.detach()
        q_primes = self.comupte_q(phi_s_primes, a_primes)

        q_prime = q_primes.mean(0)
        r = rs.mean(0)
        q1 = r + self.discount * q_prime

        q = lam * q0 + (1 - lam) * q1

        return q, r

class EnsembleDeterministicNet(nn.Module, BaseNet):
    def __init__(self, actor_body, critic_body, action_dim, num_actors, gpu=-1):
        super(EnsembleDeterministicNet, self).__init__()
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.action_dim = action_dim
        self.num_actors = num_actors

        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 3e-3)
        self.fc_actors = layer_init(nn.Linear(actor_body.feature_dim, action_dim * num_actors))
        self.set_gpu(gpu)

    def actor(self, obs, to_numpy=False):
        obs = self.tensor(obs)
        phi_actor = self.actor_body(obs)
        actions = F.tanh(self.fc_actors(phi_actor)).view(-1, self.num_actors, self.action_dim)
        obs = obs.unsqueeze(1).expand(-1, actions.size(1), -1)
        q_values = self.critic(obs, actions).squeeze(-1)
        best = q_values.max(1)[1]
        if to_numpy:
            actions = actions[self.tensor(np.arange(actions.size(0))).long(), best, :]
            return actions.detach().cpu().numpy()
        return actions, q_values, best

    def critic(self, obs, action):
        obs = self.tensor(obs)
        action = self.tensor(action)
        return self.fc_critic(self.critic_body(obs, action))

    def zero_critic_grad(self):
        self.critic_body.zero_grad()
        self.fc_critic.zero_grad()

class ThinDeterministicOptionCriticNet(nn.Module, BaseNet):
    def __init__(self, actor_body, critic_body, action_dim, num_options, gpu=-1):
        super(ThinDeterministicOptionCriticNet, self).__init__()
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.action_dim = action_dim
        self.num_options = num_options

        self.fc_actors = nn.ModuleList([layer_init(nn.Linear(
            actor_body.feature_dim, action_dim)) for _ in range(num_options)])
        self.fc_critics = nn.ModuleList([layer_init(nn.Linear(
            critic_body.feature_dim, 1)) for _ in range(num_options)])
        self.set_gpu(gpu)

    def actor(self, obs, to_numpy=False):
        obs = self.tensor(obs)
        phi_actor = self.actor_body(obs)
        actions = [F.tanh(fc_actor(phi_actor)) for fc_actor in self.fc_actors]
        q_values = [fc_critic(self.critic_body(obs, action))
                    for fc_critic, action in zip(self.fc_critics, actions)]
        q_values = torch.cat(q_values, dim=1)
        actions = torch.stack(actions).transpose(0, 1)
        best = q_values.max(1)[1]
        if to_numpy:
            actions = actions[self.tensor(np.arange(actions.size(0))).long(), best, :]
            return actions.detach().cpu().numpy(), best.detach().cpu().numpy()
        return actions, q_values, best

    def critic(self, obs, action):
        obs = self.tensor(obs)
        action = self.tensor(action)
        phi = self.critic_body(obs, action)
        q = [fc_critic(phi) for fc_critic in self.fc_critics]
        q = torch.cat(q, dim=1)
        return q

    def zero_critic_grad(self):
        self.critic_body.zero_grad()
        self.fc_critics.zero_grad()

class PlanEnsembleDeterministicNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_actors, discount, detach_action, gpu=-1):
        super(PlanEnsembleDeterministicNet, self).__init__()
        self.body = body
        phi_dim = body.feature_dim
        self.discount = discount
        self.detach_action = detach_action
        self.q_model = CriticModel(phi_dim, action_dim)
        self.action_models = nn.ModuleList([ActorModel(phi_dim, action_dim) for _ in range(num_actors)])
        self.env_model = EnvModel(phi_dim, action_dim)
        self.num_actors = num_actors
        self.set_gpu(gpu)

    def predict(self, obs, depth, to_numpy=False):
        phi = self.compute_phi(obs)
        actions = self.compute_a(phi, detach=True)
        q_values = [self.compute_q(phi, action, depth) for action in actions]
        q_values = torch.stack(q_values).squeeze(-1).t()
        actions = torch.stack(actions).t()
        if to_numpy:
            best = q_values.max(1)[1]
            actions = actions[self.tensor(np.arange(actions.size(0))).long(), best, :]
            return actions.detach().cpu().numpy()
        return q_values.max(1)[0].unsqueeze(-1)

    def compute_phi(self, obs):
        obs = self.tensor(obs)
        return self.body(obs)

    def compute_a(self, phi, detach):
        actions = [action_model(phi) for action_model in self.action_models]
        if detach:
            for action in actions: action.detach_()
        return actions

    def compute_q(self, phi, action, depth=1, immediate_reward=False):
        if depth == 1:
            q = self.q_model(phi, action)
            if immediate_reward:
                return q, 0
            return q
        else:
            phi_prime, r = self.env_model(phi, action)
            a_prime = self.compute_a(phi_prime, self.detach_action)
            a_prime = torch.stack(a_prime)
            phi_prime = phi_prime.unsqueeze(0).expand(
                (self.num_actors, ) + (-1, ) * len(phi_prime.size())
            )
            q_prime = self.compute_q(phi_prime, a_prime, depth - 1)
            q_prime = q_prime.max(0)[0]
            q = r + self.discount * q_prime
            if immediate_reward:
                return q, r
            return q

    def actor(self, obs):
        phi = self.compute_phi(obs)
        actions = self.compute_a(phi, detach=False)
        return actions
