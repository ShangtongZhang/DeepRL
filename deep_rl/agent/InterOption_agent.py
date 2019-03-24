#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class InterOptionAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.option_net = config.network_fn()
        state_dict = torch.load(config.saved_option_net, map_location=lambda storage, loc: storage)
        self.option_net.load_state_dict(state_dict)

        self.total_steps = 0
        self.worker_index = tensor(np.arange(config.num_workers)).long()

        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def compose_pi_hat(self, prediction, prev_option, is_intial_states, epsilon):
        config = self.config
        if config.control_type == 'q':
            q_option = prediction['q']
            inter_pi = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
            greedy_option = q_option.argmax(dim=-1, keepdim=True)
            prob = 1 - epsilon + epsilon / q_option.size(1)
            prob = torch.zeros_like(inter_pi).add(prob)
            inter_pi.scatter_(1, greedy_option, prob)
        elif config.control_type == 'pi':
            inter_pi = prediction['inter_pi']
        else:
            raise NotImplementedError

        mask = torch.zeros_like(inter_pi)
        mask[:, prev_option] = 1
        beta = prediction['beta'].detach()
        pi_hat = (1 - beta) * mask + beta * inter_pi

        is_intial_states = is_intial_states.view(-1, 1).expand(-1 ,inter_pi.size(1))
        pi_hat = torch.where(is_intial_states, inter_pi, pi_hat)
        return pi_hat

    def predict(self, states, net):
        config = self.config
        option_prediction = self.option_net(states)
        for k, v in option_prediction.items():
            option_prediction[k] = v.detach()
        if config.verify:
            return option_prediction

        if config.pretrained_phi:
            prediction = net(self.states, phi=option_prediction['phi'])
        else:
            prediction = net(self.states)

        return {
            'q': prediction['q'],
            'beta': option_prediction['beta'],
            'log_pi': option_prediction['log_pi'],
            'pi': option_prediction['pi'],
            'log_inter_pi': prediction['log_inter_pi'],
            'inter_pi': prediction['inter_pi'],
        }

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'inter_pi',
                                                  'log_inter_pi', 'pi_hat', 'ent_pi_hat', 'all_pi'])
        for _ in range(config.rollout_length):
            prediction = self.predict(self.states, self.network)
            epsilon = config.random_option_prob(config.num_workers)
            if config.verify: epsilon = 0.1
            pi_hat = self.compose_pi_hat(prediction, self.prev_options, self.is_initial_states, epsilon)
            dist = torch.distributions.Categorical(probs=pi_hat)
            options = dist.sample()
            ent_pi_hat = dist.entropy()

            all_pi = prediction['pi']
            prediction['pi'] = prediction['pi'][self.worker_index, options]
            dist = torch.distributions.Categorical(probs=prediction['pi'])
            actions = dist.sample()

            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         'a': actions.unsqueeze(-1),
                         'init': self.is_initial_states.unsqueeze(-1).float(),
                         'pi_hat': pi_hat,
                         'ent_pi_hat': ent_pi_hat.unsqueeze(-1),
                         'all_pi': all_pi,
                         })

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        if config.verify: return

        with torch.no_grad():
            prediction = self.predict(self.states, self.target_network)
            storage.placeholder()
            betas = prediction['beta'][self.worker_index, self.prev_options]
            ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
                  betas * (prediction['q'] * prediction['inter_pi']).mean(-1)
            ret = ret.unsqueeze(-1)

        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            storage.ret[i] = ret

        q, ret, option, action, initial_states, prev_o, pi_hat, ent_pi_hat, all_pi = \
            storage.cat(['q', 'ret', 'o', 'a', 'init', 'prev_o', 'pi_hat', 'ent_pi_hat', 'all_pi'])

        q_o = q.gather(1, option)
        v_hat = (q * pi_hat).mean(-1).unsqueeze(-1)
        adv_hat = (q_o - v_hat).detach()
        if config.pi_hat_grad == 'sample':
            pi_hat_loss = -pi_hat.add(1e-5).log().gather(1, option) * adv_hat - config.ent_hat * ent_pi_hat
        elif config.pi_hat_grad == 'expected':
            pi_hat_loss = -(pi_hat * q.detach()).sum(-1) - config.ent_hat * ent_pi_hat
        elif config.pi_hat_grad == 'posterior':
            pi_a = all_pi.gather(-1, action.unsqueeze(-1).expand(-1, pi_hat.size(1), -1))
            post = pi_hat * pi_a.squeeze(-1)
            post = post / post.sum(-1).unsqueeze(-1)
            post = post.detach()
            pi_hat_loss = -(pi_hat.add(1e-5).log() * q.detach() * post).sum(-1) - config.ent_hat * ent_pi_hat
        else:
            raise NotImplementedError
        pi_hat_loss = pi_hat_loss.mean()

        q_loss = (q_o - ret.detach()).pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        (pi_hat_loss + q_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
