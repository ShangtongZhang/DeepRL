#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import gym
import sys
from dqn_agent import *

class BasicTask:
    def transfer_state(self, state):
        return state

    def reset(self):
        return self.transfer_state(self.env.reset())

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.transfer_state(next_state)
        return next_state, reward, done, info

class MountainCar(BasicTask):
    state_space_size = 2
    action_space_size = 3
    name = 'MountainCar-v0'
    success_threshold = -110
    discount = 0.99
    step_limit = 5000
    target_network_update_freq = 1000

    def __init__(self):
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize

        self.network_fn = lambda learning_rate=0.01: FullyConnectedNet([self.state_space_size, 50, 200, self.action_space_size], learning_rate)
        self.policy_fn = lambda: GreedyPolicy(epsilon=0.5, decay_factor=0.95, min_epsilon=0.1)
        self.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)

class CartPole(BasicTask):
    state_space_size = 4
    action_space_size = 2
    name = 'CartPole-v0'
    success_threshold = 195
    discount = 0.99
    step_limit = 5000
    target_network_update_freq = 200

    def __init__(self):
        self.env = gym.make(self.name)

        self.network_fn = lambda learning_rate=0.01: FullyConnectedNet([self.state_space_size, 50, 200, self.action_space_size], learning_rate)
        self.policy_fn = lambda: GreedyPolicy(epsilon=0.5, decay_factor=0.95, min_epsilon=0.1)
        self.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)

if __name__ == '__main__':
    task = MountainCar()
    bp_network_fn = lambda learning_rate=0.001: FullyConnectedNet([task.state_space_size, 50, 200, task.action_space_size], learning_rate, gpu=False)
    def smd_network_fn(learning_rate=0.001):
        bp_network = bp_network_fn(learning_rate)
        return SMDNetworkWrapper(bp_network)

    agent = DQNAgent(task, smd_network_fn, task.policy_fn, task.replay_fn,
                     task.discount, task.step_limit, task.target_network_update_freq)
    window_size = 100
    ep = 0
    rewards = []
    while True:
        ep += 1
        reward = agent.episode()
        rewards.append(reward)
        if len(rewards) > window_size:
            reward = np.mean(rewards[-window_size:])
        print 'episode %d: %f' % (ep, reward)
        if reward > task.success_threshold:
            break
