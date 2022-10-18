#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from os import rmdir
from cv2 import Tonemap
from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class OffPACAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.state = None
        self.step_inside_ep = 0

    def eval_step(self, state):
        prediction = self.network(state)
        mu = self.behavior_policy(prediction, on_policy=True)
        return to_np(mu['action'])
    
    def behavior_policy(self, prediction, on_policy):
        config = self.config
        if on_policy:
            dist = torch.distributions.Categorical(logits=prediction['logits'])
            action = dist.sample()
        else:
            prob = torch.softmax(prediction['logits'].mul(config.mu_temperature), -1)
            prob = config.eps_mu * prob + (1 - config.eps_mu) * 1.0 / prob.size(1)
            dist = torch.distributions.Categorical(probs=prob)
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return dict(
            action=action,
            log_prob=log_prob
        )

    def step(self):
        config = self.config
        if self.state is None:
            self.state = self.task.reset()
            self.step_inside_ep = 0

        prediction = self.network(self.state)
        mu = self.behavior_policy(prediction, config.on_policy)
        action = mu['action']
        next_state, reward, done, info = self.task.step(to_np(action))
        self.record_online_return(info)
        self.state = next_state
        self.total_steps += 1

        offset = 100
        t = float(self.total_steps) + 1000 * offset
        actor_weight =  offset * t ** (-config.eps_actor) 
        critic_weight = (offset+1) * t ** (-config.eps_critic) 
        reg_0 = 0.025
        reg_weight = reg_0 * t ** (-config.eps_reg)

        reward = tensor(reward).unsqueeze(-1)
        mask = 1 - tensor(done).unsqueeze(-1)
        action = action.unsqueeze(-1)
        prediction_next = self.network(next_state)
        q = prediction['q'].gather(1, action)
        q_bootstrap = (prediction_next['prob'] * prediction_next['q']).sum(-1).unsqueeze(-1)
        if config.sac:
            q_bootstrap = q_bootstrap + reg_weight * prediction_next['entropy']
        q_target = reward + config.discount * mask * q_bootstrap
        q_loss = (q_target.detach() - q).pow(2).mul(0.5).mean()
        pi_log_prob = prediction['log_prob'].gather(1, action)
        if config.on_policy:
            rho = 1
        else:
            rho = (pi_log_prob - mu['log_prob']).exp()

        r_max = tensor(1 / (1 - config.discount))
        if config.sac:
            r_max = r_max + reg_0 * math.log(2) / (1 - config.discount)

        if config.sac:
            q = prediction['q'] - reg_weight * prediction['log_prob']
            q = torch.clamp(q, -r_max, r_max)
            pi_loss = -(prediction['prob'] * q.detach()).sum(-1)
        else:
            pi_loss = -pi_log_prob * (rho * torch.clamp(q, -r_max, r_max)).detach()
            if config.reg == 'kl':
                pi_loss = pi_loss + reg_weight * prediction['kl']
            elif config.reg == 'ent':
                pi_loss = pi_loss - reg_weight * prediction['entropy']
            else:
                raise NotImplementedError

        if config.on_policy and config.discount_state:
            pi_loss = config.discount ** self.step_inside_ep * pi_loss

        pi_loss = pi_loss.mean()
 
        loss = actor_weight * pi_loss + critic_weight * q_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.add_scalar('q_max', prediction['q'].abs().max(), self.total_steps, log_level=5)
        self.logger.add_scalar('ent_max', prediction['entropy'].max(), self.total_steps, log_level=5)
        self.logger.add_scalar('actor_w', actor_weight, self.total_steps, log_level=4)
        self.logger.add_scalar('critic_w', critic_weight, self.total_steps, log_level=4)
        self.logger.add_scalar('reg_w', reg_weight, self.total_steps, log_level=4)

        self.step_inside_ep += 1
        if done[0]:
            self.step_inside_ep = 0
