#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class TargetNetAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.total_steps = 0
        self.state = None

    def soft_update(self, target_net, main_net):
        for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.lr_t) +
                               main_param * self.config.lr_t)

    def step(self):
        config = self.config
        if self.state is None:
            self.state = self.task.reset()

        env = self.task.env.envs[0].env
        if config.game == 'BairdPrediction-v0':
            action_info = env.act()
        elif config.game == 'BairdControl-v0':
            if config.softmax_mu:
                q = self.target_network(env.expand_phi(self.state[0])).flatten()
                prob = torch.softmax(q, dim=-1)
                prob = to_np(prob)
                epsilon = 0.9
                action_info = env.act(epsilon * 6.0 / 7 + (1 - epsilon) * prob[0])
            else:
                action_info = env.act()
        else:
            raise NotImplementedError
        action = action_info['action']

        next_state, reward, done, info = self.task.step([action])
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)
        reward = tensor(reward).view(1, -1)

        if config.game == 'BairdPrediction-v0':
            target = reward + config.discount * self.target_network(next_state)
            target = target.detach()
            rho = action_info['pi_prob'] / action_info['mu_prob']
            loss = (target - self.network(self.state)).pow(2).mul(0.5).mul(rho).mean() + \
                   config.ridge * self.network.ridge().mean()
        elif config.game == 'BairdControl-v0':
            target = self.target_network(env.expand_phi(next_state[0]))
            target = reward + config.discount * target.max().view(1, 1)
            target = target.detach()
            q = self.network(env.expand_phi(self.state[0]))
            loss = (target - q[[action]]).pow(2).mul(0.5).mean() + \
                   config.ridge * self.network.ridge().mean()
        else:
            raise NotImplementedError

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.soft_update(self.target_network, self.network)

        self.state = next_state
        self.total_steps += 1

    def eval_episodes(self):
        config = self.config
        env = self.task.env.envs[0].env
        # for i in range(config.state_dim):
        #     self.logger.add_scalar('weight %d' % (i), self.network.fc.weight.data.flatten()[i])
        self.logger.add_scalar('weight', self.network.fc.weight.data.flatten().norm())
        if config.game == 'BairdPrediction-v0':
            phi = env.phi
        elif config.game == 'BairdControl-v0':
            phi = np.concatenate([env.expand_phi(phi) for phi in env.phi], axis=0)
        else:
            raise NotImplementedError
        error = self.network(phi).flatten().norm()
        # print(error)
        self.logger.add_scalar('error', error)
