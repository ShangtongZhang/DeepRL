#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
from scipy.stats import norm

class ContinuousReverseTDAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.state = self.task.reset()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0

        self.prev_gamma = 0
        self.G = 0

        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)
        self.cumulative_density = self.cumulative_density.unsqueeze(-1)

        self.bellman_error = None

    def step(self):
        config = self.config
        action, rho = config.policy(self.state)
        next_state, reward, done, info = self.task.step(action)
        self.state = next_state

        gamma = self.prev_gamma
        self.prev_gamma = 0 if done else 0.999

        self.G = reward[0] + gamma * self.G

        if config.frozen and self.total_steps > int(1e4):
            if config.mutation == 1:
                # self.G += np.random.choice([0, -5])
                self.G += np.random.choice([0, config.mutation_meta])
            elif config.mutation == 2:
                # config.mu = 0.9
                config.mu = config.mutation_meta
            else:
                raise NotImplementedError

        self.replay.feed([self.state[0], rho[0], next_state[0], reward[0], gamma])

        prediction = self.network(next_state)
        quantiles = to_np(prediction['v']).flatten()

        print(self.G, quantiles[0], quantiles[-1])
        # prob = (1 if self.G < quantiles[0] or self.G > quantiles[-1] else 0)
        # mean_q = np.mean(quantiles)
        # std_q = np.std(quantiles)
        # deviation = np.abs(self.G - mean_q) / std_q
        # if deviation > 3:
        #     prob = 99.7
        # elif deviation > 2:
        #     prob = 95
        # elif deviation > 1:
        #     prob = 68
        # else:
        #     prob = 0

        delta = 1
        std = delta
        probs = []
        for quantile in quantiles:
            r = norm.cdf(self.G + delta, quantile, std)
            l = norm.cdf(self.G - delta, quantile, std)
            probs.append(r - l)
        prob = 1 - np.mean(probs)

        if config.frozen and self.total_steps % (config.max_steps // 100) == 0:
            self.logger.add_scalar('prob_outlier', prob)
            self.logger.add_scalar('G', self.G)

        if not config.frozen and self.replay.size() >= config.warm_up:
            state, rho, next_state, reward, gamma_s = self.replay.sample()

            prediction = self.target_network(state)
            prediction_next = self.network(next_state)
            rho = tensor(rho).unsqueeze(-1)
            reward = tensor(reward).unsqueeze(-1)
            gamma_s = tensor(gamma_s).unsqueeze(-1)

            samples = reward + gamma_s * prediction['v']
            quantiles = prediction_next['v']

            self.bellman_error = (samples.mean(1) - quantiles.mean(1)).pow(2).mul(0.5).mean()
            self.logger.add_scalar('bellman_error', self.bellman_error, log_level=3)

            samples = samples.detach().unsqueeze(1)
            quantiles = quantiles.unsqueeze(-1)
            diff = samples - quantiles
            loss = rho.unsqueeze(-1) * huber(diff, 1) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
            loss = loss.mean()

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.opt.step()

            self.logger.add_scalar('qr_loss', loss)

        self.total_steps += 1

        if self.total_steps // self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def eval_episodes(self):
        pass
