#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import *

class OneStepQLearning:
    def __init__(self, config, learning_network, target_network):
        self.config = config
        self.optimizer = config.optimizer_fn(learning_network.parameters())
        self.worker_network = config.network_fn()
        self.worker_network.load_state_dict(learning_network.state_dict())
        self.task = config.task_fn()
        self.policy = config.policy_fn()
        self.learning_network = learning_network
        self.target_network = target_network

    def episode(self, deterministic=False):
        config = self.config
        state = self.task.reset()
        steps = 0
        total_reward = 0
        pending = []
        while not config.stop_signal.value:
            q = self.worker_network.predict(np.stack([state]))
            action = self.policy.sample(q.data.numpy().flatten(), deterministic)
            next_state, reward, terminal, _ = self.task.step(action)

            steps += 1
            total_reward += reward
            reward = config.reward_shift_fn(reward)

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
                    q_next, _ = self.target_network.predict(np.stack([next_state])).data.max(1)
                    if terminal and i == len(pending) - 1:
                        q_next = torch.FloatTensor([[0]])
                    q_next = config.discount * q_next + reward
                    q = q.gather(1, Variable(torch.LongTensor([[action]]))).unsqueeze(1)
                    loss += 0.5 * (q - Variable(q_next)).pow(2)

                pending = []
                self.worker_network.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.worker_network.parameters(), config.gradient_clip)
                sync_grad(self.learning_network, self.worker_network)
                self.optimizer.step()
                self.worker_network.load_state_dict(self.learning_network.state_dict())
                self.worker_network.reset(terminal)

            if terminal:
                break
            state = next_state

            if config.total_steps.value % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())

        return steps, total_reward