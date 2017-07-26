#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

class AdvantageActorCritic:
    def __init__(self, config):
        self.config = config
        self.optimizer = config.optimizer_fn(config.learning_network.parameters())
        self.worker_network = config.network_fn()
        self.worker_network.load_state_dict(config.learning_network.state_dict())
        self.task = config.task_fn()
        self.policy = config.policy_fn()

    def episode(self, deterministic=False):
        config = self.config
        state = self.task.reset()
        steps = 0
        total_reward = 0
        pending = []
        while not config.stop_signal.value and \
                (not config.max_episode_length or steps < config.max_episode_length):
            prob, log_prob, value = self.worker_network.predict(np.stack([state]))
            action = self.policy.sample(prob.data.numpy().flatten(), deterministic)
            next_state, reward, terminal, _ = self.task.step(action)

            steps += 1
            total_reward += reward

            if deterministic:
                if terminal:
                    break
                state = next_state
                continue

            pending.append([prob, log_prob, value, action, reward])
            with config.steps_lock:
                config.total_steps.value += 1

            if terminal or len(pending) >= config.update_interval:
                loss = 0
                if terminal:
                    R = torch.FloatTensor([[0]])
                else:
                    R = self.worker_network.critic(np.stack([next_state])).data
                GAE = torch.FloatTensor([[0]])
                for i in reversed(range(len(pending))):
                    prob, log_prob, value, action, reward = pending[i]
                    R = reward + config.discount * R
                    advantage = Variable(R) - value
                    GAE = config.discount * GAE + advantage.data
                    loss += 0.5 * advantage.pow(2)
                    loss += -log_prob.gather(1, Variable(torch.LongTensor([[action]]))) * Variable(GAE)
                    loss += 0.01 * torch.sum(torch.mul(prob, log_prob))

                pending = []
                self.worker_network.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.worker_network.parameters(), config.gradient_clip)
                self.optimizer.zero_grad()
                for param, worker_param in zip(
                        config.learning_network.parameters(), self.worker_network.parameters()):
                    param._grad = worker_param.grad.clone()
                self.optimizer.step()
                self.worker_network.load_state_dict(config.learning_network.state_dict())
                self.worker_network.reset(terminal)

            if terminal:
                break
            state = next_state

        return steps, total_reward