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
        pi = F.softmax(pi, dim=-1)
        return {'q': q,
                'beta': beta,
                'log_pi': log_pi,
                'pi': pi}


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
        
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
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
        
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
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
        
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}


class TD3Net(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 actor_body_fn,
                 critic_body_fn,
                 actor_opt_fn,
                 critic_opt_fn,
                 ):
        super(TD3Net, self).__init__()
        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()

        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        obs = tensor(obs)
        return torch.tanh(self.fc_action(self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2

class OptionLstmGaussianActorCriticNet(BaseNet):

  def __init__(self,
               state_dim,
               action_dim,
               hid_dim,
               num_options,
               phi_body=None,
               actor_body=None,
               critic_body=None,
               option_body_fn=None,
               config=None):
    super().__init__(config)
    self.is_recur = True
    self.config = config
    self.action_dim = action_dim
    self.num_options = num_options

    self.hid_dim = hid_dim
    if config.bi_direction:
      # h,c: (num_layers * num_directions, batch, hidden_size)
      self.hid_size = [config.num_lstm_layers * 2, None, self.hid_dim]
    else:
      self.hid_size = [config.num_lstm_layers, None, self.hid_dim]

    if phi_body is None:
      phi_body = DummyBody(state_dim)
    if critic_body is None:
      critic_body = DummyBody(config.lstm_to_fc_feat_dim)
    if actor_body is None:
      actor_body = DummyBody(config.lstm_to_fc_feat_dim)

    self.phi_body = phi_body
    self.actor_body = actor_body
    self.critic_body = critic_body

    self.lstm = lstm_init(
        nn.LSTM(
            phi_body.feature_dim,
            hid_dim,
            num_layers=config.num_lstm_layers,
            dropout=config.lstm_dropout,
            bidirectional=config.bi_direction), 1e-3)

    self.fc_pi_o = layer_init(
        nn.Linear(config.lstm_to_fc_feat_dim, action_dim), 1e-3)
    self.fc_q_o = layer_init(nn.Linear(config.lstm_to_fc_feat_dim, 1), 1e-3)

    # build option network
    self.options = nn.ModuleList([
        SingleOptionLstmNet(action_dim, hid_dim, option_body_fn, config)
        for _ in range(num_options)
    ])

    # this is Config module, not self.config
    # device is selected select_device() in main
    self.to(Config.DEVICE)

  def forward(self,
              obs,
              masks,
              prev_options,
              input_manager_lstm_states=None,
              input_options_lstm_states_list=None):
    '''
    Parameter:
      obs: [timesteps, batch, feat_dim]
      input_lstm_states: (h, c) h/c: [num_layers * num_directions, batch, hidden_size]
      masks: [timesteps, batch]
      prev_options: [timesteps, batch]
      input_option_lstm_states_list: [num_options]: tuple(input_option_lstm_states)

    Returns:
      'input_lstm_states': [num_layers * num_directions, batch, hidden_size]
      'final_lstm_states': [num_layers * num_directions, batch, hidden_size]
      'a': [timesteps * batchsize, action_dim]
      'log_pi_a': [timesteps * batchsize, 1]
      'h_lstm_states': [timesteps * batchsize, hid_dim * num_layers * num_directions]
      'ent': [timesteps * batchsize, 1]
      'mean': [timesteps * batchsize, action_dim]
      'v': [timesteps * batchsize, 1]
    '''
    obs = tensor(obs)
    batch_size = masks.shape[1]

    masks = tensor(masks)
    # extends to [timesteps, batch, 1]
    # so it can be broadcast to [num_layers * num_directions, batch, hidden_size]
    # when multiplied with h and c
    masks = masks.unsqueeze(-1)

    if not input_manager_lstm_states:
      input_manager_lstm_states = self.get_init_lstm_states(batch_size)

    phi = self.phi_body(obs)

    # LSTM Loop
    h_list = []
    h_input, c_input = input_manager_lstm_states
    for p, m in zip(phi, masks):
      h_input = h_input * m
      c_input = c_input * m
      _, final_manager_lstm_states = self.lstm(
          p.unsqueeze(0),
          (h_input, c_input))
      h_input, c_input = final_manager_lstm_states
      # h,c: (num_layers * num_directions, batch, hidden_size)
      h_list.append(h_input)

    # output (fc) layers loop
    pi_o_list = []
    log_pi_o_list = []
    q_o_list = []
    for h in h_list:
      # flat h into [batch, feat_dim] shape (ffn's input)
      # h: (num_layers * num_directions, batch, hidden_size)->
      #    (batch, hidden_size * num_layers * num_directions)
      h = h.permute([1, 0, 2]).reshape([batch_size, -1])

      # policy over option with soft-max
      phi_a = self.actor_body(h)
      pi_o = F.softmax(phi_a, dim=-1)
      log_pi_o = F.log_softmax(phi_a, dim=-1)

      # critic network
      phi_c = self.critic_body(phi)
      q_o = self.fc_q_o(phi_c)

      pi_o_list.append(pi_o)
      log_pi_o_list.append(log_pi_o)
      q_o_list.append(q_o)

    pi_o, log_pi_o, q_o = [
        torch.cat(i) for i in [pi_o_list, log_pi_o_list, q_o_list]
    ]

    # option
    mean = []
    std = []
    beta = []
    h_lstm_states_list = []
    final_option_lstm_states_list = []

    # pause: how to deal with prev_options
    # agent: how to deal with prev_options | h_states
    aligned_input_options_lstm_states_list = self.get_aligned_input_options_lstm_states_list(
        input_options_lstm_states_list, prev_options[0])
    for o, option in enumerate(self.options):
      aligned_masks = self.get_aligned_options_masks(masks, prev_options, o)
      prediction = option(phi, aligned_masks,
                          aligned_input_options_lstm_states_list[o])
      mean.append(prediction['mean'].unsqueeze(1))
      std.append(prediction['std'].unsqueeze(1))
      beta.append(prediction['beta'])
      h_lstm_states_list.append(prediction['h_lstm_states'])
      final_option_lstm_states_list.append(prediction['final_lstm_states'])
    mean = torch.cat(mean, dim=1)
    std = torch.cat(std, dim=1)
    beta = torch.cat(beta, dim=1)

    return {
        'pi_o': pi_o,
        'log_pi_o': log_pi_o,
        'q_o': q_o,
        'final_manager_lstm_states': final_manager_lstm_states,
        'mean': mean,
        'std': std,
        'beta': beta,
        'h_lstm_states_list': h_lstm_states_list,
        'final_option_lstm_states_list': final_option_lstm_states_list
    }

  def get_init_lstm_states(self, batchsize):
    # h,c: (num_layers * num_directions, batch, hidden_size)
    self.hid_size[1] = batchsize
    init_states = (torch.zeros(self.hid_size, device=Config.DEVICE),
                   torch.zeros(self.hid_size, device=Config.DEVICE))
    return init_states

  def get_init_options_lstm_states_list(self, batchsize):
    # h,c: (num_layers * num_directions, batch, hidden_size)
    options_lstm_states_list = []
    for i in range(self.num_options):
      options_lstm_states_list.append(
          self.options[i].get_init_lstm_states(batchsize))
    return options_lstm_states_list

  def get_aligned_input_options_lstm_states_list(
      self, input_options_lstm_states_list, prev_options):
    '''
    input_option_lstm_states_list: list[num_options]; contains tuple of lstm states (h,c)
    prev_options: [batchsize]; each entry is option id selected at t-1 time
    '''
    batch_size = prev_options.shape[0]
    aligned_input_options_lstm_states_list = self.get_init_options_lstm_states_list(
        batch_size)

    if not input_options_lstm_states_list:
      return aligned_input_options_lstm_states_list

    for batch_id, option_id in enumerate(prev_options):
      ih, ic = input_options_lstm_states_list[option_id]
      ah, ac = aligned_input_options_lstm_states_list[option_id]
      ah[:, batch_id, :] = ih[:, batch_id, :]
      ac[:, batch_id, :] = ic[:, batch_id, :]

    return aligned_input_options_lstm_states_list

  def get_aligned_options_masks(self, masks, prev_options, option_id):
    '''
    masks: [timesteps, batch]
    prev_options: [timesteps, batch]
    '''
    aligned_masks = masks.copy()
    for m, o in zip(aligned_masks, prev_options):
      o = o == option_id
      m = m * o
      import ipdb
      ipdb.set_trace(context=7)
    return aligned_masks


class SingleOptionLstmNet(BaseNet):

  def __init__(self, action_dim, hid_dim, phi_body_fn, config=None):
    super().__init__(config)
    self.is_recur = True
    self.config = config
    self.action_dim = action_dim
    self.hid_dim = hid_dim

    self.phi_body = phi_body_fn()

    if config.bi_direction:
      # h,c: (num_layers * num_directions, batch, hidden_size)
      self.hid_size = [config.num_lstm_layers * 2, None, self.hid_dim]
    else:
      self.hid_size = [config.num_lstm_layers, None, self.hid_dim]

    self.lstm = lstm_init(
        nn.LSTM(
            self.phi_body.feature_dim,
            hid_dim,
            num_layers=config.num_lstm_layers,
            dropout=config.lstm_dropout,
            bidirectional=config.bi_direction), 1e-3)

    self.fc_mean = layer_init(
        nn.Linear(config.lstm_to_fc_feat_dim, action_dim), 1e-3)
    self.fc_beta = layer_init(nn.Linear(config.lstm_to_fc_feat_dim, 1), 1e-3)

    # this is Config module, not self.config
    # device is selected select_device() in main
    self.std = nn.Parameter(torch.zeros(action_dim))
    self.to(Config.DEVICE)

  def forward(self, obs, masks, input_lstm_states=None):
    '''
    Parameter:
      obs: [timesteps, batch, feat_dim]
      input_lstm_states: (h, c) h/c: [num_layers * num_directions, batch, hidden_size]
      masks: [timesteps, batch]

    Returns:
      'mean': [timesteps * batchsize, action_dim]
      'std': [timesteps * batchsize, action_dim]
      'beta': [timesteps * batchsize, 1]
      'final_lstm_states': [num_layers * num_directions, batch, hidden_size]
    '''
    obs = tensor(obs)
    batch_size = masks.shape[1]

    masks = tensor(masks)
    # extends to [timesteps, batch, 1]
    # so it can be broadcast to [num_layers * num_directions, batch, hidden_size]
    # when multiplied with h and c
    masks = masks.unsqueeze(-1)

    if not input_lstm_states:
      input_lstm_states = self.get_init_lstm_states(batch_size)

    phi = self.phi_body(obs)

    # LSTM Loop
    h_list = []
    h_input, c_input = input_lstm_states
    for p, m in zip(phi, masks):
      h_input = h_input * m
      c_input = c_input * m
      _, final_lstm_states = self.lstm(p.unsqueeze(0), (h_input, c_input))
      h_input, c_input = final_lstm_states
      # h,c: (num_layers * num_directions, batch, hidden_size)
      h_list.append(h_input)

    # output (fc) layers loop
    mean_list = []
    beta_list = []
    h_out_list = []
    for h in h_list:
      # flat h into [batch, feat_dim] shape (ffn's input)
      # h: (num_layers * num_directions, batch, hidden_size)->
      #    (batch, hidden_size * num_layers * num_directions)
      h = h.permute([1, 0, 2]).reshape([batch_size, -1])
      h_out_list.append(h)

      mean = torch.tanh(self.fc_pi(h))
      beta = F.softmax(self.fc_beta(h), dim=-1)

      mean_list.append(mean)
      beta_list.append(beta)

    mean, beta, h_out = [
        torch.cat(i) for i in [mean_list, beta_list, h_out_list]
    ]
    std = F.softplus(self.std).expand(mean.size(0), -1)

    return {
        'mean': mean,
        'beta': beta,
        'std': std,
        'h_lstm_states': h_out,
        'final_lstm_states': final_lstm_states
    }

  def get_init_lstm_states(self, batchsize):
    # h,c: (num_layers * num_directions, batch, hidden_size)
    self.hid_size[1] = batchsize
    init_states = (torch.zeros(self.hid_size, device=Config.DEVICE),
                   torch.zeros(self.hid_size, device=Config.DEVICE))
    return init_states


