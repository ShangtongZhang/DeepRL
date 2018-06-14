#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *

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

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 gpu=-1):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.set_gpu(gpu)

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        action = self.actor(phi)
        if to_numpy:
            return action.cpu().detach().numpy()
        return action

    def feature(self, obs):
        obs = self.tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 gpu=-1):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.set_gpu(gpu)

    def predict(self, obs, action=None, to_numpy=False):
        obs = self.tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, self.tensor(np.zeros((log_prob.size(0), 1))), v

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 gpu=-1):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.set_gpu(gpu)

    def predict(self, obs, action=None):
        obs = self.tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        prob = F.softmax(self.network.fc_action(phi_a), dim=-1)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(probs=prob)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v

class ThinDeterministicOptionCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 phi_body,
                 actor_body,
                 critic_body,
                 num_options,
                 actor_opt_fn,
                 critic_opt_fn,
                 gpu=-1):
        super(ThinDeterministicOptionCriticNet, self).__init__()
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.action_dim = action_dim
        self.num_options = num_options

        self.fc_actors = nn.ModuleList([layer_init(nn.Linear(
            actor_body.feature_dim, action_dim)) for _ in range(num_options)])
        self.fc_critics = nn.ModuleList([layer_init(nn.Linear(
            critic_body.feature_dim, 1)) for _ in range(num_options)])

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_actors.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critics.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)

        self.set_gpu(gpu)

    def feature(self, obs):
        obs = self.tensor(obs)
        phi = self.phi_body(obs)
        return phi

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        actions = self.actor(phi)
        q_values = self.critic(phi, actions)
        actions = torch.stack(actions).transpose(0, 1)
        best = q_values.max(1)[1]
        if to_numpy:
            actions = actions[self.tensor(np.arange(actions.size(0))).long(), best, :]
            return actions.detach().cpu().numpy(), best.detach().cpu().numpy()
        return actions, q_values, best

    def actor(self, phi):
        phi_actor = self.actor_body(phi)
        actions = [F.tanh(fc_actor(phi_actor)) for fc_actor in self.fc_actors]
        return actions

    def critic(self, phi, actions):
        if isinstance(actions, torch.Tensor):
            phi = self.critic_body(phi, actions)
            q = [fc_critic(phi) for fc_critic in self.fc_critics]
        elif isinstance(actions, list):
            q = [fc_critic(self.critic_body(phi, action))
                 for fc_critic, action in zip(self.fc_critics, actions)]
        q = torch.cat(q, dim=1)
        return q

class DeterministicOptionCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 phi_body,
                 actor_body,
                 critic_body,
                 beta_body,
                 num_options,
                 actor_opt_fn,
                 critic_opt_fn,
                 gpu=-1):
        super(DeterministicOptionCriticNet, self).__init__()
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.beta_body = beta_body
        self.action_dim = action_dim
        self.num_options = num_options

        self.fc_actors = nn.ModuleList([layer_init(nn.Linear(
            actor_body.feature_dim, action_dim)) for _ in range(num_options)])
        self.fc_beta = layer_init(nn.Linear(beta_body.feature_dim, num_options))
        self.fc_critics = nn.ModuleList([layer_init(nn.Linear(
            critic_body.feature_dim, 1)) for _ in range(num_options)])

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_actors.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critics.parameters())
        self.beta_params = list(self.beta_body.parameters()) + list(self.fc_beta.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params +
                                        self.beta_params)

        self.set_gpu(gpu)

    def feature(self, obs):
        obs = self.tensor(obs)
        phi = self.phi_body(obs)
        return phi

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        actions = self.actor(phi)
        betas = self.termination(phi)
        q_values = self.critic(phi, actions)
        actions = torch.stack(actions).transpose(0, 1)
        return q_values, betas, actions

    def termination(self, phi):
        phi_beta = self.beta_body(phi)
        beta = F.sigmoid(self.fc_beta(phi_beta))
        return beta

    def actor(self, phi):
        phi_actor = self.actor_body(phi)
        actions = [F.tanh(fc_actor(phi_actor)) for fc_actor in self.fc_actors]
        return actions

    def critic(self, phi, actions):
        if isinstance(actions, torch.Tensor):
            phi = self.critic_body(phi, actions)
            q = [fc_critic(phi) for fc_critic in self.fc_critics]
        elif isinstance(actions, list):
            q = [fc_critic(self.critic_body(phi, action))
                 for fc_critic, action in zip(self.fc_critics, actions)]
        q = torch.cat(q, dim=1)
        return q

class GammaDeterministicOptionCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 phi_body,
                 actor_body,
                 critic_body,
                 num_options,
                 gpu=-1):
        super(GammaDeterministicOptionCriticNet, self).__init__()
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.action_dim = action_dim
        self.num_options = num_options

        self.fc_q_options = layer_init(nn.Linear(actor_body.feature_dim, num_options))
        self.fc_actors = nn.ModuleList([layer_init(nn.Linear(
            actor_body.feature_dim, action_dim)) for _ in range(num_options)])
        self.fc_critics = nn.ModuleList([layer_init(nn.Linear(
            critic_body.feature_dim, 1)) for _ in range(num_options)])

        self.set_gpu(gpu)

    def feature(self, obs):
        obs = self.tensor(obs)
        phi = self.phi_body(obs)
        return phi

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        actions, q_options = self.actor(phi)
        # q_values = self.critic(phi, actions)
        # actions = torch.stack(actions).transpose(0, 1)
        # best = q_values.max(1)[1]
        # if to_numpy:
        #     actions = actions[self.tensor(np.arange(actions.size(0))).long(), best, :]
        #     return actions.detach().cpu().numpy(), best.detach().cpu().numpy()
        # return actions, q_values, best
        return actions, q_options

    def actor(self, phi):
        phi_actor = self.actor_body(phi)
        actions = [F.tanh(fc_actor(phi_actor)) for fc_actor in self.fc_actors]
        q_options = self.fc_q_options(phi_actor)
        return actions, q_options

    def critic(self, phi, actions):
        if isinstance(actions, torch.Tensor):
            phi = self.critic_body(phi, actions)
            q = [fc_critic(phi) for fc_critic in self.fc_critics]
        elif isinstance(actions, list):
            q = [fc_critic(self.critic_body(phi, action))
                 for fc_critic, action in zip(self.fc_critics, actions)]
        q = torch.cat(q, dim=-1)
        return q

    def zero_non_actor_grad(self):
        self.fc_q_options.zero_grad()
        self.fc_critics.zero_grad()
        self.critic_body.zero_grad()

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

class PlanEnsembleDeterministicNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body,
                 num_actors,
                 discount,
                 detach_action,
                 gpu=-1):
        super(PlanEnsembleDeterministicNet, self).__init__()
        self.phi_body = phi_body
        phi_dim = phi_body.feature_dim
        self.critic_model = CriticModel(phi_dim, action_dim)
        self.actor_models = nn.ModuleList([ActorModel(phi_dim, action_dim) for _ in range(num_actors)])
        self.env_model = EnvModel(phi_dim, action_dim)

        self.discount = discount
        self.detach_action = detach_action
        self.num_actors = num_actors
        self.set_gpu(gpu)

    def predict(self, obs, depth, to_numpy=False):
        phi = self.feature(obs)
        actions = self.compute_a(phi)
        q_values = [self.compute_q(phi, action, depth) for action in actions]
        q_values = torch.stack(q_values).squeeze(-1).t()
        actions = torch.stack(actions).t()
        if to_numpy:
            best = q_values.max(1)[1]
            actions = actions[self.range(actions.size(0)), best, :]
            return actions.detach().cpu().numpy()
        return q_values.max(1)[0].unsqueeze(-1)

    def feature(self, obs):
        obs = self.tensor(obs)
        return self.phi_body(obs)

    def compute_a(self, phi, detach=True):
        actions = [actor_model(phi) for actor_model in self.actor_models]
        if detach:
            for action in actions: action.detach_()
        return actions

    def compute_q(self, phi, action, depth=1, immediate_reward=False):
        if depth == 1:
            q = self.critic_model(phi, action)
            if immediate_reward:
                return q, 0
            return q
        else:
            phi_prime, r = self.env_model(phi, action)
            a_prime = self.compute_a(phi_prime)
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

class QuantileDDPGNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 num_quantiles,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 gpu=-1):
        super(QuantileDDPGNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, num_quantiles), 1e-3)
        self.num_quantiles = num_quantiles

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.set_gpu(gpu)

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        action = self.actor(phi)
        if to_numpy:
            return action.cpu().detach().numpy()
        return action

    def feature(self, obs):
        obs = self.tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))

class QuantileEnsembleDDPGNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 num_quantiles,
                 num_actors,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 gpu=-1):
        super(QuantileEnsembleDDPGNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim * num_actors), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, num_quantiles), 1e-3)
        self.num_quantiles = num_quantiles
        self.num_actors = num_actors
        self.action_dim = action_dim

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.set_gpu(gpu)

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        actions = self.actor(phi)
        q_values = self.critic(phi.unsqueeze(1).expand(-1, actions.size(1), -1), actions)
        if to_numpy:
            q_values = q_values.mean(-1)
            best = torch.argmax(q_values, dim=-1)
            best_action = actions[self.range(actions.size(0)), best, :]
            return best_action.cpu().detach().numpy()
        return q_values

    def feature(self, obs):
        obs = self.tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.fc_action(self.actor_body(phi))).view(-1, self.num_actors, self.action_dim)

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))