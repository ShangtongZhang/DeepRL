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
    def __init__(self, agent):
        self.agent = agent
        self.optimizer = agent.optimizer_fn(agent.learning_network.parameters())
        self.worker_network = agent.network_fn()
        self.worker_network.load_state_dict(agent.learning_network.state_dict())
        self.task = agent.task_fn()
        self.policy = agent.policy_fn()

    def episode(self):
        state = self.task.reset()
        steps = 0
        total_reward = 0
        pending = []
        while True and not self.agent.stop_signal.value:
            prob, log_prob, value = self.worker_network.predict(np.stack([state]))
            action = self.policy.sample(prob.data.numpy().flatten())
            next_state, reward, terminal, _ = self.task.step(action)
            pending.append([prob, log_prob, value, action, reward])

            steps += 1
            with self.agent.steps_lock:
                self.agent.total_steps.value += 1
            total_reward += reward

            if terminal or len(pending) >= self.agent.update_interval:
                loss = 0
                if terminal:
                    R = torch.FloatTensor([[0]])
                else:
                    R = self.worker_network.critic(np.stack([next_state])).data
                for i in reversed(range(len(pending))):
                    prob, log_prob, value, action, reward = pending[i]
                    R = reward + self.agent.discount * R
                    advantage = Variable(R) - value
                    loss += 0.5 * advantage.pow(2)
                    loss += -log_prob.gather(1, Variable(torch.LongTensor([[action]]))) * Variable(advantage.data)
                    loss += 0.01 * torch.sum(torch.mul(prob, log_prob))

                pending = []
                self.worker_network.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.worker_network.parameters(), 40)
                self.optimizer.zero_grad()
                for param, worker_param in zip(
                        self.agent.learning_network.parameters(), self.worker_network.parameters()):
                    param._grad = worker_param.grad.clone()
                self.optimizer.step()
                self.worker_network.load_state_dict(self.agent.learning_network.state_dict())
                self.worker_network.reset(terminal)

            if terminal:
                break
            else:
                state = next_state

        return steps, total_reward

class NStepQLearning:
    def __init__(self, agent):
        self.agent = agent
        self.optimizer = agent.optimizer_fn(agent.learning_network.parameters())
        self.worker_network = agent.network_fn()
        self.worker_network.load_state_dict(agent.learning_network.state_dict())
        self.task = agent.task_fn()
        self.policy = agent.policy_fn()

    def episode(self):
        state = self.task.reset()
        steps = 0
        total_reward = 0
        pending = []
        while True and not self.agent.stop_signal.value:
            q = self.worker_network.predict(np.stack([state]))
            action = self.policy.sample(q.data.numpy().flatten())
            next_state, reward, terminal, _ = self.task.step(action)
            pending.append([q, action, reward])

            steps += 1
            with self.agent.steps_lock:
                self.agent.total_steps.value += 1
            total_reward += reward

            if terminal or len(pending) >= self.agent.update_interval:
                loss = 0
                if terminal:
                    R = torch.FloatTensor([[0]])
                else:
                    R, _ = self.agent.target_network.predict(
                        np.stack([next_state])).data.max(1)

                for i in reversed(range(len(pending))):
                    q, action, reward = pending[i]
                    R = reward + self.agent.discount * R
                    loss += 0.5 * (Variable(R) - q.gather(1, Variable(torch.LongTensor([[action]])))).pow(2)

                pending = []
                self.worker_network.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.worker_network.parameters(), 40)
                self.optimizer.zero_grad()
                for param, worker_param in zip(
                        self.agent.learning_network.parameters(), self.worker_network.parameters()):
                    param._grad = worker_param.grad.clone()
                self.optimizer.step()
                self.worker_network.load_state_dict(self.agent.learning_network.state_dict())
                self.worker_network.reset(terminal)

            if terminal:
                break
            else:
                state = next_state

            if self.agent.total_steps.value % self.agent.target_network_update_freq == 0:
                self.agent.target_network.load_state_dict(
                    self.agent.learning_network.state_dict())

        return steps, total_reward

class OneStepQLearning:
    def __init__(self, agent):
        self.agent = agent
        self.optimizer = agent.optimizer_fn(agent.learning_network.parameters())
        self.worker_network = agent.network_fn()
        self.worker_network.load_state_dict(agent.learning_network.state_dict())
        self.task = agent.task_fn()
        self.policy = agent.policy_fn()

    def episode(self):
        state = self.task.reset()
        steps = 0
        total_reward = 0
        pending = []
        while True and not self.agent.stop_signal.value:
            q = self.worker_network.predict(np.stack([state]))
            action = self.policy.sample(q.data.numpy().flatten())
            next_state, reward, terminal, _ = self.task.step(action)
            pending.append([q, action, reward, next_state])

            steps += 1
            with self.agent.steps_lock:
                self.agent.total_steps.value += 1
            total_reward += reward

            if terminal or len(pending) >= self.agent.update_interval:
                loss = 0
                for i in range(len(pending)):
                    q, action, reward, next_state = pending[i]
                    q_next, _ = self.agent.target_network.predict(np.stack([next_state])).data.max(1)
                    if terminal and i == len(pending) - 1:
                        q_next = torch.FloatTensor([[0]])
                    q_next = self.agent.discount * q_next + reward
                    q = q.gather(1, Variable(torch.LongTensor([[action]])))
                    loss += 0.5 * (q - Variable(q_next)).pow(2)

                pending = []
                self.worker_network.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.worker_network.parameters(), 40)
                self.optimizer.zero_grad()
                for param, worker_param in zip(
                        self.agent.learning_network.parameters(), self.worker_network.parameters()):
                    param._grad = worker_param.grad.clone()
                self.optimizer.step()
                self.worker_network.load_state_dict(self.agent.learning_network.state_dict())
                self.worker_network.reset(terminal)

            if terminal:
                break
            else:
                state = next_state

            if self.agent.total_steps.value % self.agent.target_network_update_freq == 0:
                self.agent.target_network.load_state_dict(
                    self.agent.learning_network.state_dict())

        return steps, total_reward

class OneStepSarsa:
    def __init__(self, agent):
        self.agent = agent
        self.optimizer = agent.optimizer_fn(agent.learning_network.parameters())
        self.worker_network = agent.network_fn()
        self.worker_network.load_state_dict(agent.learning_network.state_dict())
        self.task = agent.task_fn()
        self.policy = agent.policy_fn()

    def episode(self):
        state = self.task.reset()
        q = self.worker_network.predict(np.stack([state]))
        action = self.policy.sample(q.data.numpy().flatten())
        steps = 0
        total_reward = 0
        pending = []
        while True and not self.agent.stop_signal.value:
            next_state, reward, terminal, _ = self.task.step(action)
            next_q = self.worker_network.predict(np.stack([next_state]))
            next_action = self.policy.sample(next_q.data.numpy().flatten())
            pending.append([q, action, reward, next_state, next_action])

            steps += 1
            with self.agent.steps_lock:
                self.agent.total_steps.value += 1
            total_reward += reward

            if terminal or len(pending) >= self.agent.update_interval:
                loss = 0
                for i in range(len(pending)):
                    q, action, reward, next_state, next_action = pending[i]
                    q_next = self.agent.target_network.predict(np.stack([next_state])).data
                    if terminal and i == len(pending) - 1:
                        q_next = torch.FloatTensor([[0]])
                    else:
                        q_next = q_next.gather(1, torch.LongTensor([[next_action]]))
                    q_next = self.agent.discount * q_next + reward
                    q = q.gather(1, Variable(torch.LongTensor([[action]])))
                    loss += 0.5 * (q - Variable(q_next)).pow(2)

                pending = []
                self.worker_network.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.worker_network.parameters(), 40)
                self.optimizer.zero_grad()
                for param, worker_param in zip(
                        self.agent.learning_network.parameters(), self.worker_network.parameters()):
                    param._grad = worker_param.grad.clone()
                self.optimizer.step()
                self.worker_network.load_state_dict(self.agent.learning_network.state_dict())
                self.worker_network.reset(terminal)

            if terminal:
                break
            else:
                q = next_q
                action = next_action

            if self.agent.total_steps.value % self.agent.target_network_update_freq == 0:
                self.agent.target_network.load_state_dict(
                    self.agent.learning_network.state_dict())

        return steps, total_reward
