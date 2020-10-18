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


class ReverseTDAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.state = self.task.reset()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0

        self.prev_gamma = 0

        self.G = 0
        self.G_oracle = 0

        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)
        self.cumulative_density = self.cumulative_density.unsqueeze(-1)

        self.bellman_error = None
        if 'Robot' in config.game:
            self.compute_oracle()

    def simulate_trajectory(self, max_steps):
        config = self.config
        env = self.task
        G = 0
        Gs = []
        env.reset()
        for _ in range(max_steps):
            action = [np.random.choice([0, 1], p=[1 - config.mu, config.mu])]
            _, reward, _, info = env.step(action)
            info = info[0]
            G = reward[0] + info['gamma_s'] * G
            Gs.append([info['next_s'], G])
        return Gs

    def simulate_oracle(self):
        NUM_STATES = 4
        runs = 1000
        max_steps = 100
        data = dict()
        for s in range(NUM_STATES):
            for t in range(max_steps):
                data[(s, t)] = []
        for _ in range(runs):
            Gs = self.simulate_trajectory(max_steps)
            for t in range(max_steps):
                s, G = Gs[t]
                data[(s, t)].append(G)
        mean = np.zeros((NUM_STATES, max_steps))
        std = np.zeros(mean.shape)
        for s in range(NUM_STATES):
            for t in range(max_steps):
                mean[s, t] = np.mean(data[(s, t)])
                std[s, t] = np.std(data[(s, t)])
        state_to_plot = 0
        # import matplotlib.pyplot as plt
        # plt.errorbar(x=np.arange(mean.shape[1]), y=mean[state_to_plot], yerr=std[state_to_plot])
        # plt.ylim([0, 10])
        # plt.show()
        print(np.mean(mean[:, -10:], axis=1))
        return data

    def compute_oracle(self):
        NUM_STATES = 4
        PI = self.config.pi
        prob_success = 0.99
        P_pi = np.zeros((NUM_STATES, NUM_STATES))
        P_pi[0, 0] = P_pi[1, 1] = P_pi[2, 2] = P_pi[3, 3] = 1 - prob_success
        P_pi[0, 1] = P_pi[1, 2] = P_pi[2, 3] = P_pi[3, 0] = PI * prob_success
        P_pi[0, 3] = P_pi[1, 0] = P_pi[2, 1] = P_pi[3, 2] = (1 - PI) * prob_success
        gamma = np.zeros(P_pi.shape)
        gamma[0, 0] = gamma[1, 1] = gamma[2, 2] = 1
        P_tilde = np.zeros((NUM_STATES * 2, NUM_STATES))
        r = np.zeros((NUM_STATES * 2, 1))
        for i in range(NUM_STATES):
            P_tilde[2 * i, i] = P_tilde[2 * i + 1, i] = 1 - prob_success
            P_tilde[2 * i, (i + 1) % NUM_STATES] = 1 * prob_success
            P_tilde[2 * i + 1, (i - 1 + NUM_STATES) % NUM_STATES] = 1 * prob_success
            r[2 * i, 0] = 2 * prob_success
            r[2 * i + 1, 0] = 1 * prob_success

        P = np.eye(NUM_STATES)
        P_star = 0
        N = 1000
        for _ in range(N):
            P_star += P
            P = P @ P_pi
        P_star = P_star / float(N)
        d = P_star[0, :]
        D_pi = np.diag(d)
        D_tilde = np.zeros((NUM_STATES * 2, NUM_STATES * 2))
        for i in range(NUM_STATES):
            D_tilde[2 * i, 2 * i] = d[i] * PI
            D_tilde[2 * i + 1, 2 * i + 1] = d[i] * (1 - PI)

        v_bar = np.linalg.inv(D_pi) @ np.linalg.inv(np.eye(NUM_STATES) - P_pi.T @ gamma) \
                @ P_tilde.T @ D_tilde @ r
        self.v_bar_oracle = tensor(v_bar)
        print(v_bar)

    def step(self):
        config = self.config
        action, rho = config.policy(self.state)
        next_state, reward, done, info = self.task.step(action)

        prediction = self.network(self.state)
        prediction_next = self.network(next_state)
        rho = tensor(rho).unsqueeze(-1)
        reward = tensor(reward).unsqueeze(-1)
        gamma = self.prev_gamma
        self.prev_gamma = [unit['gamma_next_s'] for unit in info]
        self.prev_gamma = tensor(self.prev_gamma).unsqueeze(-1)

        G_s = self.G
        self.G = reward + gamma * self.G

        if config.loss_type == 'td':
            target = reward + gamma * ((1 - config.lam) * prediction['v'] + config.lam * G_s)
            target = target.detach()
            loss = (prediction_next['v'] - target).mul(rho).pow(2).mul(0.5).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.bellman_error = loss

        elif config.loss_type == 'qr':
            samples = reward + gamma * prediction['v']
            quantiles = prediction_next['v']

            self.bellman_error = (samples.mean(1) - quantiles.mean(1)).pow(2).mul(0.5).mean()
            self.logger.add_scalar('bellman_error', self.bellman_error, log_level=3)
            # print(prediction['v'])

            if not config.frozen:
                samples = samples.detach().unsqueeze(1)
                quantiles = quantiles.unsqueeze(-1)
                diff = samples - quantiles
                loss = rho * huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
                loss = loss.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.opt.step()
            else:
                if self.total_steps > int(1e4):
                    if config.mutation == 1:
                        self.G += np.random.choice([0, 2])
                    elif config.mutation == 2:
                        config.mu = 0.9
                    else:
                        raise NotImplementedError

            quantiles = to_np(quantiles).flatten()
            delta = 1
            std = 1
            probs = []
            for quantile in quantiles:
                r = norm.cdf(self.G + delta, quantile, std)
                l = norm.cdf(self.G - delta, quantile, std)
                probs.append(r - l)
            prob = 1 - np.mean(probs)

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

            if config.frozen and self.total_steps % (config.max_steps // 100) == 0:
                self.logger.add_scalar('prob_outlier', prob)
                self.logger.add_scalar('G', self.G)

        elif config.loss_type == 'mc':
            loss = (self.G.detach() - prediction_next['v']).pow(2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        else:
            raise NotImplementedError

        # next_s = info[0]['next_s']
        # self.G_oracle = self.v_bar_oracle[next_s, 0]
        # G_loss = np.abs(self.G - self.G_oracle)
        # print(G_loss)
        # self.logger.add_scalar('G_loss', G_loss)

        self.state = next_state
        self.total_steps += 1

    def eval_episodes(self):
        config = self.config
        if 'Robot' in config.game:
            phi = config.eval_env.env.envs[0].env.phi
            prediction = self.network(tensor(phi))
            if config.loss_type in ['td', 'mc']:
                v = prediction['v']
            elif config.loss_type == 'qr':
                v = prediction['v'].mean(1).unsqueeze(1)
            else:
                raise NotImplementedError
            loss = (v - self.v_bar_oracle).pow(2).sum().pow(0.5)
            print('v_bar_loss: %s' % (loss))
            self.logger.add_scalar('v_bar_loss', loss)

