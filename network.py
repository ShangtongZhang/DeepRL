#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import tensorflow as tf
from common import *

class Network:
    def __init__(self, name, dim_in, dim_out, optimizer_fn, initializer=tf.random_normal_initializer()):
        self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
        dim_hidden1 = 50
        dim_hidden2 = 200
        W1, b1, net1, phi1 = \
            fully_connected(name, 'layer1', self.x, dim_in, dim_hidden1, initializer, tf.nn.relu)
        W2, b2, net2, phi2 = \
            fully_connected(name, 'layer2', phi1, dim_hidden1, dim_hidden2, initializer, tf.nn.relu)
        W3, b3, net3, self.y = \
            fully_connected(name, 'layer3', phi2, dim_hidden2, dim_out, initializer, tf.identity)
        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))
        loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.y, self.target))
        self.variables = [W1, b1, W2, b2, W3, b3]
        self.train_op = optimizer_fn(name).minimize(loss=loss)

    def get_assign_ops(self, src_network):
        assign_ops = []
        for dst_var, src_var in zip(self.variables, src_network.variables):
            assign_ops.append(dst_var.assign(src_var))
        return assign_ops

    def predict(self, sess, x):
        y = sess.run(self.y, feed_dict={self.x: x})
        return y

    def learn(self, sess, x, target):
        sess.run(self.train_op, feed_dict={self.x: x, self.target: target})

class SimpleNetwork(Network):
    def __init__(self, name, dim_in, dim_out, dim_hidden, optimizer_fn, initializer=tf.random_normal_initializer()):
        self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
        W1, b1, net1, phi1 = \
            fully_connected(name, 'layer1', self.x, dim_in, dim_hidden, initializer, tf.nn.relu)
        W2, b2, net2, self.y = \
            fully_connected(name, 'layer2', phi1, dim_hidden, dim_out, initializer, tf.identity)
        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))
        loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.y, self.target))
        self.variables = [W1, b1, W2, b2]
        self.train_op = optimizer_fn(name).minimize(loss=loss)
