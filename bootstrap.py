#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from torch.autograd import Variable

class OneStepSarsa:
    def __init__(self, agent):
        self.agent = agent
        self.reset()

    def reset(self):
        self.pending = []

    def process_state(self, network, state):
        q = network.predict(np.stack([state]))
        self.pending.append([q])
        return q.data.numpy().flatten()

    def process_interaction(self, action, reward, next_state):
        self.pending[-1].extend([action, reward, next_state])

    def compute_loss(self, network, terminal):
        loss = 0
        valid_length = len(self.pending)
        if not terminal:
            valid_length -= 1
        for i in range(valid_length):
            q, action, reward, next_state = self.pending[i]
            q_next = self.agent.target_network.predict(np.stack([next_state])).data
            if i < len(self.pending) - 1:
                next_action = self.pending[i + 1][1]
                q_next = q_next.gather(1, torch.LongTensor([[next_action]]))
            else:
                q_next = torch.FloatTensor([[0]])
            q_next = self.agent.discount * q_next + reward
            q = q.gather(1, Variable(torch.LongTensor([[action]])))
            loss += 0.5 * (q - Variable(q_next)).pow(2)
        self.reset()
        return loss


class OneStepQLearning:
    def __init__(self, agent):
        self.agent = agent
        self.reset()

    def reset(self):
        self.pending = []

    def process_state(self, network, state):
        q = network.predict(np.stack([state]))
        self.pending.append([q])
        return q.data.numpy().flatten()

    def process_interaction(self, action, reward, next_state):
        self.pending[-1].extend([action, reward, next_state])

    def compute_loss(self, network, terminal):
        loss = 0
        for i in range(len(self.pending)):
            q, action, reward, next_state = self.pending[i]
            q_next, _ = self.agent.target_network.predict(np.stack([next_state])).data.max(1)
            if terminal and i == len(self.pending) - 1:
                q_next = torch.FloatTensor([[0]])
            q_next = self.agent.discount * q_next + reward
            q = q.gather(1, Variable(torch.LongTensor([[action]])))
            loss += 0.5 * (q - Variable(q_next)).pow(2)
        self.reset()
        return loss

class NStepQLearning:
    def __init__(self, agent):
        self.agent = agent
        self.reset()

    def reset(self):
        self.pending = []

    def process_state(self, network, state):
        q = network.predict(np.stack([state]))
        self.pending.append([q])
        return q.data.numpy().flatten()

    def process_interaction(self, action, reward, next_state):
        self.pending[-1].extend([action, reward])
        self.tailing_state = next_state

    def compute_loss(self, network, terminal):
        loss = 0
        if terminal:
            R = torch.FloatTensor([[0]])
        else:
            R, _ = self.agent.target_network.predict(
                np.stack([self.tailing_state])).data.max(1)

        for i in reversed(range(len(self.pending))):
            q, action, reward = self.pending[i]
            R = reward + self.agent.discount * R
            loss += 0.5 * (Variable(R) - q.gather(1, Variable(torch.LongTensor([[action]])))).pow(2)
        self.reset()
        return loss

class AdvantageActorCritic:
    def __init__(self, agent):
        self.agent = agent
        self.reset()

    def reset(self):
        self.pending = []

    def process_state(self, network, state):
        prob, log_prob, value = network.predict(np.stack([state]))
        self.pending.append([prob, log_prob, value])
        return prob.data.numpy().flatten()

    def process_interaction(self, action, reward, next_state):
        self.pending[-1].extend([action, reward])
        self.tailing_state = next_state

    def compute_loss(self, network, terminal):
        loss = 0
        if terminal:
            R = torch.FloatTensor([[0]])
        else:
            R = network.critic(np.stack([self.tailing_state])).data
        for i in reversed(range(len(self.pending))):
            prob, log_prob, value, action, reward = self.pending[i]
            R = reward + self.agent.discount * R
            advantage = Variable(R) - value
            loss += 0.5 * advantage.pow(2)
            loss += -log_prob.gather(1, Variable(torch.LongTensor([[action]]))) * Variable(advantage.data)
            loss += 0.01 * torch.sum(torch.mul(prob, log_prob))
        self.reset()
        return loss

