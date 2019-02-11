#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
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
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
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
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}

class EnsembleDeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 num_actors,
                 num_critics,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(EnsembleDeterministicActorCriticNet, self).__init__()
        # self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim * num_actors), 1e-3)
        self.fc_critic = nn.Linear(critic_body.feature_dim, num_critics)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.to(Config.DEVICE)

        self.action_dim = action_dim
        self.num_actors = num_actors
        self.num_critics = num_critics

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        actions = F.tanh(self.fc_action(self.actor_body(phi)))
        actions = actions.view(-1, self.num_actors, self.action_dim)
        return actions

    def critic(self, phi, a):
        q = self.fc_critic(self.critic_body(phi, a))
        return q

class Model(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 ensemble_size,
                 p_hidden_units=512,
                 r_hidden_units=128,
                 type='D'):
        super(Model, self).__init__()
        assert type in ['D', 'P']

        self.fc_body_p = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, p_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(p_hidden_units, p_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(p_hidden_units, p_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(p_hidden_units, p_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(p_hidden_units, p_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(p_hidden_units, p_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(p_hidden_units, p_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(p_hidden_units, p_hidden_units)),
            nn.ReLU(),
        )

        self.fc_body_r = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, r_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(r_hidden_units, r_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(r_hidden_units, r_hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(r_hidden_units, r_hidden_units)),
            nn.ReLU(),
        )

        self.fc_p_mean = layer_init(nn.Linear(p_hidden_units, state_dim * ensemble_size))
        if type == 'P':
            self.fc_p_std = layer_init(nn.Linear(p_hidden_units, state_dim * ensemble_size))

        self.fc_r = layer_init(nn.Linear(r_hidden_units, ensemble_size))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.type = type
        self.to(Config.DEVICE)

    def transition(self, s_a):
        phi_p = self.fc_body_p(s_a)
        mean = self.fc_p_mean(phi_p)

        phi_r = self.fc_body_r(s_a)
        r = self.fc_r(phi_r)

        mean = mean.view(-1, self.ensemble_size, self.state_dim)
        if self.type == 'P':
            std = F.softplus(self.fc_p_std(phi_p))
            std = std.view(-1, self.ensemble_size, self.state_dim)
        else:
            std = None
        return mean, std, r

    def loss(self, s, a, r, next_s):
        s_a = torch.cat([s, a], dim=-1)
        mean, std, r_hat = self.transition(s_a)
        r_loss = (r - r_hat).pow(2).mul(0.5)
        next_s = next_s.unsqueeze(1)
        if self.type == 'P':
            dist = DiagonalNormal(mean, std)
            log_prob = dist.log_prob(next_s).squeeze(-1)
            p_loss = -log_prob
        elif self.type == 'D':
            s = s.unsqueeze(1)
            delta_s = next_s - s
            p_loss = (mean - delta_s).pow(2).mul(0.5).sum(-1)
        else:
            raise NotImplementedError
        return p_loss, r_loss

    def forward(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        mean, std, r = self.transition(s_a)
        if self.type == 'P':
            next_s = mean
        elif self.type == 'D':
            next_s = s.unsqueeze(1) + mean
        else:
            raise NotImplementedError
        return r, next_s


class BackwardModel(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_units,
                 ensemble_size,
                 type):
        super(BackwardModel, self).__init__()
        assert type in ['D', 'P']

        self.fc_body = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_units, hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_units, hidden_units)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_units, hidden_units)),
            nn.ReLU(),
        )

        self.fc_p_mean = layer_init(nn.Linear(hidden_units, state_dim * ensemble_size))
        if type == 'P':
            self.fc_p_std = layer_init(nn.Linear(hidden_units, state_dim * ensemble_size))

        self.fc_r = layer_init(nn.Linear(hidden_units, ensemble_size))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.type = type
        self.to(Config.DEVICE)

    def transition(self, s_a):
        phi = self.fc_body(s_a)
        r = self.fc_r(phi)
        mean = self.fc_p_mean(phi)
        mean = mean.view(-1, self.ensemble_size, self.state_dim)
        if self.type == 'P':
            std = F.softplus(self.fc_p_std(phi))
            std = std.view(-1, self.ensemble_size, self.state_dim)
        else:
            std = None
        return mean, std, r

    def loss(self, s, a, r, next_s):
        s_a = torch.cat([next_s, a], dim=-1)
        mean, std, r_hat = self.transition(s_a)
        r_loss = (r - r_hat).pow(2).mul(0.5)
        s = s.unsqueeze(1)
        if self.type == 'P':
            raise NotImplementedError
        elif self.type == 'D':
            next_s = next_s.unsqueeze(1)
            delta_s = next_s - s
            p_loss = (mean - delta_s).pow(2).mul(0.5).sum(-1)
        else:
            raise NotImplementedError
        return p_loss, r_loss

    def forward(self, s, a):
        s = tensor(s)
        a = tensor(a)
        s_a = torch.cat([s, a], dim=-1)
        mean, std, r = self.transition(s_a)
        if self.type == 'P':
            prev_s = mean
        elif self.type == 'D':
            prev_s = s.unsqueeze(1) - mean
        else:
            raise NotImplementedError
        return r, prev_s

