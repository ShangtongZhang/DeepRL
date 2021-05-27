#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class NeuralOPEDataGenerator(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.network = config.network_fn()
        self.load('./data/DifferentialGQ/%s-policy' % config.game)
        self.data = []

    def collect_data(self):
        config = self.config
        self.sample_trajectory()
        s, a, r, next_s = list(map(lambda x: np.concatenate(x, axis=0), zip(*self.data)))
        next_a = self.network(tensor(next_s))
        sa = np.concatenate([s, a], axis=1)
        next_sa = np.concatenate([next_s, to_np(next_a)], axis=1)
        data = [sa, r.reshape(-1, 1), next_sa]
        rate = self.simulate_oracle_rate()
        print(rate)
        dataset = dict(data=data, rate=rate)
        with open('./data/DifferentialGQ/%s.pkl' % (config.dataset), 'wb') as f:
            pickle.dump(dataset, f)

    def simulate_oracle_rate(self):
        config = self.config
        env = config.eval_env
        states = env.reset()
        all_rewards = []
        for i in range(config.rate_samples):
            actions = self.network(states)
            next_states, rewards, done, info = env.step(to_np(actions))
            all_rewards.append(rewards)
            states = next_states
            print('computing oracle %d' % (i))
        return np.mean(all_rewards)

    def sample_trajectory(self):
        config = self.config
        env = config.eval_env
        states = env.reset()
        for i in range(config.data_samples):
            actions = to_np(self.sample_action(tensor(states), config.noise))
            next_states, rewards, done, info = env.step(actions)
            self.data.append([states, actions, rewards, next_states])
            states = next_states
            print('sampling %d' % (i))

    def sample_action(self, states, std):
        actions = self.network(states)
        actions += torch.randn(actions.size()) * std
        return actions


class NeuralOPEAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.discount = 1
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_net = config.network_fn()
        self.target_net.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.GQ2_cache = None
        self.rates = deque(maxlen=100)
        self.load_dataset()

    def load_dataset(self):
        config = self.config
        with open('./data/DifferentialGQ/%s.pkl' % (config.dataset), 'rb') as f:
            dataset = pickle.load(f)
        self.data = dataset['data']
        sa, r, next_sa = self.data
        self.data = dict(sa=tensor(sa), r=tensor(r), next_sa=tensor(next_sa))
        self.rate_oracle = dataset['rate']

    def sample(self):
        config = self.config
        indices = np.random.randint(0, self.data['sa'].size(0), size=config.batch_size)
        return self.data['sa'][indices], self.data['r'][indices], self.data['next_sa'][indices]

    def evaluation(self):
        config = self.config
        rate = np.mean(self.rates)
        if np.isnan(rate):
            rate = np.asscalar(to_np(self.network.rate()))
        rate_loss = np.abs(self.rate_oracle - rate)
        self.logger.add_scalar('online_rate_loss', rate_loss)
        sa, r, next_sa = self.data['sa'], self.data['r'], self.data['next_sa']
        if config.algo in ['GenDICE', 'GradientDICE']:
            tau = self.network.tau(sa)
            rate = (tau * r).mean()
        elif config.algo in ['FQE', 'FQE-target', 'GQ1', 'GQ2']:
            rate = r + self.network.v(next_sa) - self.network.v(sa)
            rate = rate.mean()
        else:
            raise NotImplementedError
        rate = np.asscalar(to_np(rate))
        rate_loss = np.abs(self.rate_oracle - rate)
        self.logger.add_scalar('offline_rate_loss', rate_loss)

    def step(self):
        loss = None
        config = self.config
        sa, r, next_sa = self.sample()
        if config.algo in ['GenDICE', 'GradientDICE']:
            tau = self.network.tau(sa)
            f = self.network.f(sa)
            f_next = self.network.f(next_sa)
            u = self.network.u(sa.size(0))
            rate = self.network.rate()
            if config.algo == 'GenDICE':
                J_concave = config.discount * tau.detach() * f_next - \
                            tau.detach() * (f + 0.25 * f.pow(2)) + config.lam * (u * tau.detach() - u - 0.5 * u.pow(2))
                J_convex = config.discount * tau * f_next.detach() - \
                           tau * (f.detach() + 0.25 * f.detach().pow(2)) + \
                           config.lam * (u.detach() * tau - u.detach() - 0.5 * u.detach().pow(2))
            elif config.algo == 'GradientDICE':
                J_concave = config.discount * tau.detach() * f_next - \
                            tau.detach() * f - 0.5 * f.pow(2) + config.lam * (u * tau.detach() - u - 0.5 * u.pow(2))
                J_convex = config.discount * tau * f_next.detach() - \
                           tau * f.detach() - 0.5 * f.detach().pow(2) + \
                           config.lam * (u.detach() * tau - u.detach() - 0.5 * u.detach().pow(2))
            else:
                raise NotImplementedError
            r_loss = (tau.detach().mul(r).mean() - rate).pow(2).mul(0.5).mean()
            J_convex = J_convex.mean() + r_loss
            loss = J_convex - J_concave.mean()
        elif config.algo == 'FQE':
            rate = self.network.rate()
            v_target = r - rate + self.network.v(next_sa)
            v_loss = (v_target.detach() - self.network.v(sa)).pow(2).mul(0.5).mean()
            rate_target = r + self.network.v(next_sa) - self.network.v(sa)
            rate_loss = (rate_target.detach().mean() - rate).pow(2).mul(0.5).mean()
            loss = v_loss + rate_loss
        elif config.algo == 'FQE-target':
            rate = self.network.rate()
            rate_t = self.target_net.rate()
            v_t = self.target_net.v(sa)
            next_v_t = self.target_net.v(next_sa)
            v_loss = (r - rate_t + next_v_t - self.network.v(sa)).pow(2).mul(0.5).mean()
            rate_loss = ((r + next_v_t - v_t).mean() - rate).pow(2).mul(0.5).mean()
            loss = v_loss + rate_loss
        elif config.algo == 'GQ1':
            rate = self.network.rate()
            v = self.network.v(sa)
            next_v = self.network.v(next_sa)
            nu = self.network.nu(sa)
            td_error = r - rate + next_v - v
            J_concave = 2 * nu * td_error.detach() - nu.pow(2)
            J_convex = 2 * nu.detach() * td_error
            loss = J_convex.mean() - J_concave.mean()
            rate = self.network.rate()
        elif config.algo == 'GQ2':
            if self.GQ2_cache is None:
                self.GQ2_cache = [sa, r, next_sa]
            else:
                sa1, r1, next_sa1 = self.GQ2_cache
                rate1 = r1 + self.network.v(next_sa1) - self.network.v(sa1)
                rate2 = r + self.network.v(next_sa) - self.network.v(sa)
                td_error = rate2 - rate1
                nu = self.network.nu(sa)
                J_concave = 2 * nu * td_error.detach() - nu.pow(2)
                J_convex = 2 * nu.detach() * td_error
                rate = self.network.rate()
                r_loss = ((rate1 + rate2).mul(0.5).detach().mean() - rate).pow(2).mul(0.5).mean()
                loss = J_convex.mean() + r_loss - J_concave.mean()
        else:
            raise NotImplementedError

        if self.total_steps:
            if rate is None:
                rate = self.rates[-1]
            else:
                rate = np.asscalar(rate.detach().numpy())
            self.rates.append(rate)

        self.total_steps += 1
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.total_steps % self.config.target_network_update_freq == 0:
            self.target_net.load_state_dict(self.network.state_dict())

    def eval_episodes(self):
        self.evaluation()
