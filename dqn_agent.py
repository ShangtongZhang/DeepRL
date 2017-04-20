#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import tensorflow as tf
from network import *
from replay import *
from policy import *
import numpy as np

class DQNAgent:
    def __init__(self, name, task, network_fn, policy_fn, replay_fn, discount, step_limit, target_network_update_freq):
        with tf.variable_scope(name):
            self.learning_network = network_fn('learning')
            self.target_network = network_fn('target')
        self.assign_ops = self.target_network.get_assign_ops(self.learning_network)
        self.task = task
        self.step_limit = step_limit
        self.replay = replay_fn()
        self.discount = discount
        self.target_network_update_freq = target_network_update_freq
        self.policy = policy_fn()
        self.total_steps = 0

    def episode(self, sess):
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while not self.step_limit or steps < self.step_limit:
            value = self.learning_network.predict(sess, np.reshape(state, (1, -1)) )
            action = self.policy.sample(value.flatten())
            next_state, reward, done, info = self.task.step(action)
            total_reward += reward
            self.replay.feed([state, action, reward, next_state, int(done)])
            steps += 1
            self.total_steps += 1
            state = next_state
            if done:
                break
            experiences = self.replay.sample()
            if experiences is not None:
                states, actions, rewards, next_states, terminals = experiences
                targets = self.learning_network.predict(sess, states)
                q_next = self.target_network.predict(sess, next_states)
                q_next = np.max(q_next, axis=1)
                q_next = rewards + self.discount * q_next
                q_next = np.where(terminals, 0, q_next)
                targets[np.arange(len(actions)), actions] = q_next
                self.learning_network.learn(sess, states, targets)
            if self.total_steps % self.target_network_update_freq == 0:
                sess.run(self.assign_ops)
        self.policy.update_epsilon()
        return total_reward
