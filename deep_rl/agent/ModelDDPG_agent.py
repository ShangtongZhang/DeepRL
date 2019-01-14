#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision

class ModelDDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.episode_reward = 0
        self.episode_rewards = []

        self.model = config.model_fn()
        self.model_opt = config.model_opt_fn(self.model.parameters())

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def train_model(self):
        config = self.config
        for i in range(config.model_opt_epochs):
            s, a, r, next_s, _, b_mask = self.replay.sample(config.model_opt_batch_size)
            s = tensor(s).squeeze(1)
            a = tensor(a).squeeze(1)
            r = tensor(r)
            next_s = tensor(next_s).squeeze(1)
            b_mask = tensor(b_mask)

            p_loss, r_loss = self.model.loss(s, a, r, next_s)
            config.logger.add_scalar('tran_loss', p_loss.mean())
            config.logger.add_scalar('r_loss', r_loss.mean())

            loss = (p_loss * b_mask).sum(-1).mean() + (r_loss * b_mask).sum(-1).mean()

            self.model_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.model_opt.step()

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)
        if self.total_steps < config.min_memory_size:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, _ = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.episode_reward += reward[0]
        reward = self.config.reward_normalizer(reward)

        bootstrap_mask = np.random.rand(config.ensemble_size) < config.bootstrap_prob
        self.replay.feed([self.state, action, reward, next_state, done.astype(np.uint8), bootstrap_mask.astype(np.uint8)])
        if done[0]:
            config.logger.add_scalar('train_episode_return', self.episode_reward)
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.total_steps >= config.model_warm_up:
            self.train_model()

        if self.replay.size() >= config.min_memory_size:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals, _ = experiences
            states = states.squeeze(1)
            actions = actions.squeeze(1)
            rewards = tensor(rewards)
            next_states = next_states.squeeze(1)
            terminals = tensor(terminals)

            r_hat, next_s_hat = self.model(states, actions)
            next_states = torch.cat([next_s_hat, tensor(next_states).unsqueeze(1)], dim=1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            config.logger.add_scalar('q_std', q_next.std(1).mean())
            q_next = q_next.max(1)[0]
            q_next = config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, tensor(actions))
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
            config.logger.add_scalar('q_loss', critic_loss)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()
            config.logger.add_scalar('pi_loss', policy_loss)

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)
