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

        optimizer_fn = lambda name: tf.train.GradientDescentOptimizer(name=name, learning_rate=0.01)
        self.network_fn = lambda name: Network(name, self.state_space_size,
                                               self.action_space_size, optimizer_fn, tf.random_normal_initializer())
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

        optimizer_fn = lambda name: tf.train.GradientDescentOptimizer(name=name, learning_rate=0.01)
        self.network_fn = lambda name: Network(name, self.state_space_size,
                                               self.action_space_size, optimizer_fn, tf.random_normal_initializer())
        self.policy_fn = lambda: GreedyPolicy(epsilon=0.5, decay_factor=0.95, min_epsilon=0.1)
        self.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)

class LunarLander(BasicTask):
    state_space_size = 8
    action_space_size = 4
    name = 'LunarLander-v2'
    success_threshold = 200
    discount = 0.99
    step_limit = 5000
    target_network_update_freq = 200

    def __init__(self):
        self.env = gym.make(self.name)
        optimizer_fn = lambda name: tf.train.GradientDescentOptimizer(name=name, learning_rate=0.001)
        self.network_fn = lambda name: Network(name, self.state_space_size,
                                               self.action_space_size, optimizer_fn, tf.random_normal_initializer())
        self.policy_fn = lambda: GreedyPolicy(epsilon=0.5, decay_factor=0.99, min_epsilon=0.1)
        self.replay_fn = lambda: Replay(memory_size=20000, batch_size=100)

if __name__ == '__main__':
    # task = MountainCar()
    task = LunarLander()
    agent = DQNAgent(task.name, task, task.network_fn, task.policy_fn, task.replay_fn,
                     task.discount, task.step_limit, task.target_network_update_freq)
    window_size = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ep = 0
        rewards = []
        while True:
            ep += 1
            reward = agent.episode(sess)
            rewards.append(reward)
            if len(rewards) > window_size:
                reward = np.mean(rewards[-window_size:])
            print 'episode %d: %f' % (ep, reward)
            if reward > task.success_threshold:
                break