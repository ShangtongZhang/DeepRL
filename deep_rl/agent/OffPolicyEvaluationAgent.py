#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class OffPolicyEvaluation(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.DICENet = config.dice_net_fn()
        self.DICENet_target = config.dice_net_fn()
        self.DICENet_target.load_state_dict(self.DICENet.state_dict())
        self.network = config.network_fn()
        self.replay = config.replay_fn()
        self.total_steps = 0

        try:
            self.replay.load('./data/GradientDICE/%s-data%d' % (config.game, config.dataset))
        except:
            pass
        self.load('./data/GradientDICE/%s-policy' % config.game)
        self.oracle_perf = self.load_oracle_perf()
        print('True performance: %s' % (self.oracle_perf))

    def collect_data(self):
        noise = [
            [0.1],
            # [0.05, 0.1],
            # [0.1, 0.15],
            # [0.15, 0.2],
            # [0.25, 0.3]
        ]
        for std in noise[self.config.dataset - 1]:
            self.sample_trajectory(std)
        self.replay.save('./data/GradientDICE/%s-data%d' % (self.config.game, self.config.dataset))

    def sample_trajectory(self, std):
        config = self.config
        env = config.eval_env
        for i in range(1000):
            print('Sampling trajectory %s' % (i))
            states = env.reset()
            for j in range(100):
                actions = to_np(self.sample_action(tensor(states), std))
                next_states, rewards, done, info = env.step(actions)
                ret = info[0]['episodic_return']
                if ret is not None:
                    print('Episode end')
                experiences = list(zip(states, actions, rewards, next_states, done))
                self.replay.feed_batch(experiences)
                states = next_states

    def sample_action(self, states, std):
        actions = self.network(states)
        actions += torch.randn(actions.size()) * std
        return actions

    def eval_episode(self):
        config = self.config
        env = config.eval_env
        state = env.reset()
        rewards = []
        while True:
            action = to_np(self.sample_action(tensor(state), config.noise_std))
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            ret = info[0]['episodic_return']
            if ret is not None:
                print('Computing true performance: %s' % ret)
                break
        if config.discount == 1:
            return np.mean(rewards)
        ret = 0
        for r in reversed(rewards):
            ret = r + config.discount * ret
        return ret

    def load_oracle_perf(self):
        return self.compute_oracle()

    def compute_oracle(self):
        config = self.config
        if config.game == 'Reacher-v2':
            n_ep = 500
        elif config.game in ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2']:
            n_ep = 100
        else:
            raise NotImplementedError
        perf = [self.eval_episode() for _ in range(n_ep)]
        if config.discount == 1:
            return np.mean(perf)
        else:
            return (1 - config.discount) * np.mean(perf)

    def step(self):
        config = self.config
        if config.correction == 'no':
            return
        experiences = self.replay.sample()
        states, actions, rewards, next_states, terminals = experiences
        states = tensor(states)
        actions = tensor(actions)
        rewards = tensor(rewards).unsqueeze(-1)
        next_states = tensor(next_states)
        masks = tensor(1 - terminals).unsqueeze(-1)

        next_actions = self.sample_action(next_states, config.noise_std).detach()
        states_0 = tensor(config.sample_init_states())
        actions_0 = self.sample_action(states_0, config.noise_std).detach()

        tau = self.DICENet.tau(states, actions)
        f = self.DICENet.f(states, actions)
        f_next = self.DICENet.f(next_states, next_actions)
        f_0 = self.DICENet.f(states_0, actions_0)
        u = self.DICENet.u(states.size(0))

        tau_target = self.DICENet_target.tau(states, actions).detach()
        f_target = self.DICENet_target.f(states, actions).detach()
        f_next_target = self.DICENet_target.f(next_states, next_actions).detach()
        f_0_target = self.DICENet_target.f(states_0, actions_0).detach()
        u_target = self.DICENet_target.u(states.size(0)).detach()

        if config.correction == 'GenDICE':
            J_concave = (1 - config.discount) * f_0 + config.discount * tau_target * f_next - \
                        tau_target * (f + 0.25 * f.pow(2)) + config.lam * (u * tau_target - u - 0.5 * u.pow(2))
            J_convex = (1 - config.discount) * f_0_target + config.discount * tau * f_next_target - \
                       tau * (f_target + 0.25 * f_target.pow(2)) + \
                       config.lam * (u_target * tau - u_target - 0.5 * u_target.pow(2))
        elif config.correction == 'GradientDICE':
            J_concave = (1 - config.discount) * f_0 + config.discount * tau_target * f_next - \
                        tau_target * f - 0.5 * f.pow(2) + config.lam * (u * tau_target - u - 0.5 * u.pow(2))
            J_convex = (1 - config.discount) * f_0_target + config.discount * tau * f_next_target - \
                       tau * f_target - 0.5 * f_target.pow(2) + \
                       config.lam * (u_target * tau - u_target - 0.5 * u_target.pow(2))
        elif config.correction == 'DualDICE':
            J_concave = (f_target - config.discount * f_next_target) * tau - tau.pow(3).mul(1.0 / 3) \
                        - (1 - config.discount) * f_0_target
            J_convex = (f - config.discount * f_next) * tau_target - tau_target.pow(3).mul(1.0 / 3) - \
                       (1 - config.discount) * f_0
        else:
            raise NotImplementedError

        loss = (J_convex - J_concave) * masks
        self.DICENet.opt.zero_grad()
        loss.mean().backward()
        self.DICENet.opt.step()
        if self.total_steps % config.target_network_update_freq == 0:
            self.DICENet_target.load_state_dict(self.DICENet.state_dict())

        self.total_steps += 1

    def eval_episodes(self):
        experiences = self.replay.sample(len(self.replay.data))
        states, actions, rewards, next_states, terminals = experiences
        states = tensor(states)
        actions = tensor(actions)
        rewards = tensor(rewards).unsqueeze(-1)
        if self.config.correction == 'no':
            tau = 1
        else:
            tau = self.DICENet.tau(states, actions)
        perf = (tau * rewards).mean()
        loss = (perf - self.oracle_perf).pow(2).mul(0.5)
        print('perf_loss: %s' % (loss))
        self.logger.add_scalar('perf_loss', loss)
