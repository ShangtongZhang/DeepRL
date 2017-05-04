#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import tensorflow as tf

def fully_connected(model_name, layer_name, var_in, dim_in, dim_out,
                    initializer, transfer):
    with tf.variable_scope(model_name):
        with tf.variable_scope(layer_name):
            W = tf.get_variable("W", [dim_in, dim_out],
                                initializer=initializer)
            b = tf.get_variable("b", [dim_out],
                                initializer=initializer)
    net = tf.nn.bias_add(tf.matmul(var_in, W), b)
    phi = transfer(net)
    return W, b, net, phi

class Relu:
    def __init__(self):
        self.gate_fun = tf.nn.relu
        self.gate_fun_gradient = \
            lambda phi, net: tf.where(net >= 0, tf.ones(tf.shape(net)), tf.zeros(tf.shape(net)))


class Tanh:
    def __init__(self):
        self.gate_fun = tf.tanh
        self.gate_fun_gradient = \
            lambda phi, net: tf.subtract(1.0, tf.pow(phi, 2))

class Identity:
    def __init__(self):
        self.gate_fun = tf.identity
        self.gate_fun_gradient = \
            lambda phi, net: tf.ones(tf.shape(phi))

def crossprop_layer(model_name, layer_name, var_in, dim_in, dim_hidden, dim_out, gate_fun, initializer):
    with tf.variable_scope(model_name):
        with tf.variable_scope(layer_name):
            U = tf.get_variable('U', [dim_in, dim_hidden],
                                initializer=initializer)
            b_hidden = tf.get_variable('b_hidden', [dim_hidden],
                                       initializer=initializer)
            W = tf.get_variable('W', [dim_hidden, dim_out],
                                initializer=initializer)
            b_out = tf.get_variable('b_out', [dim_out],
                                    initializer=initializer)
    net = tf.matmul(var_in, U)
    net = tf.nn.bias_add(net, b_hidden)
    phi = gate_fun(net)
    y = tf.matmul(phi, W)
    y = tf.nn.bias_add(y, b_out)
    return U, b_hidden, net, phi, W, b_out, y