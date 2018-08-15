#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision

class PlanDDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        action = self.network.predict(state, depth=self.config.depth, to_numpy=True).flatten()
        self.config.state_normalizer.unset_read_only()
        return action

    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.config.state_normalizer(state)

        config = self.config

        steps = 0
        total_reward = 0.0
        while True:
            self.evaluate()
            self.evaluation_episodes()

            action, option = self.network.predict(np.stack([state]), depth=config.depth, to_numpy=True).flatten()
            action += self.random_process.sample()
            next_state, reward, done, info = self.task.step(action)
            next_state = self.config.state_normalizer(next_state)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)

            if not deterministic:
                mask = np.random.binomial(n=1, p=0.5, size=config.num_actors)
                self.replay.feed([state, action, reward, next_state, int(done), mask, option])
                self.total_steps += 1

            steps += 1
            state = next_state

            if not deterministic and self.replay.size() >= config.min_memory_size:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, masks = experiences
                masks = self.network.tensor(masks)

                q_next = self.target_network.predict(next_states, depth=config.depth, to_numpy=False)
                terminals = self.network.tensor(terminals).unsqueeze(1)
                rewards = self.network.tensor(rewards).unsqueeze(1)
                q_next = config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()

                phi = self.network.feature(states)
                q, r = self.network.compute_q(phi, self.network.tensor(actions),
                                           depth=config.depth, immediate_reward=True)
                q_loss = (q - q_next).pow(2).mul(0.5).mean()
                r_loss = (r - rewards).pow(2).mul(0.5).mean()

                self.opt.zero_grad()
                (q_loss + r_loss).mul(config.critic_loss_weight).backward()
                self.opt.step()

                phi = self.network.feature(states)
                actions = self.network.compute_a(phi, detach=False)

                dead_actions = []
                for action in actions:
                    dead_action = action.clone()
                    dead_action.detach_().requires_grad_()
                    dead_actions.append(dead_action)

                q_values = [self.network.compute_q(phi.detach(), dead_action, depth=config.depth)
                            for dead_action in dead_actions]
                q_values = torch.stack(q_values).squeeze(-1).t()
                q_values = q_values.mean(0)
                self.opt.zero_grad()
                q_values.backward(self.network.tensor(np.ones(q_values.size())))

                actions = torch.stack(actions)
                action_grads = torch.stack([-dead_action.grad.detach()
                                            for dead_action in dead_actions])
                if config.mask:
                    action_grads = action_grads * masks.t().unsqueeze(-1)
                if config.on_policy:
                    actions = 0
                self.opt.zero_grad()
                actions.backward(action_grads)
                self.opt.step()

                self.soft_update(self.target_network, self.network)

            if done:
                break

        return total_reward, steps
