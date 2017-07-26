#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

class OneStepQLearning:
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
            q = self.worker_network.predict(np.stack([state]))
            action = self.policy.sample(q.data.numpy().flatten(), deterministic)
            next_state, reward, terminal, _ = self.task.step(action)

            steps += 1
            total_reward += reward

            if deterministic:
                if terminal:
                    break
                state = next_state
                continue

            with config.steps_lock:
                config.total_steps.value += 1
            pending.append([q, action, reward, next_state])

            if terminal or len(pending) >= config.update_interval:
                loss = 0
                for i in range(len(pending)):
                    q, action, reward, next_state = pending[i]
                    q_next, _ = config.target_network.predict(np.stack([next_state])).data.max(1)
                    if terminal and i == len(pending) - 1:
                        q_next = torch.FloatTensor([[0]])
                    q_next = config.discount * q_next + reward
                    q = q.gather(1, Variable(torch.LongTensor([[action]])))
                    loss += 0.5 * (q - Variable(q_next)).pow(2)

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

            if config.total_steps.value % config.target_network_update_freq == 0:
                config.target_network.load_state_dict(config.learning_network.state_dict())

        return steps, total_reward