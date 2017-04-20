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