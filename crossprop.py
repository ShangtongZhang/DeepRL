#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import tensorflow as tf
from common import *
import numpy as np

class CrossProp:
    def __init__(self, name, dim_in, dim_hidden, dim_out, learning_rate, gate=Relu(),
                 initializer=tf.random_normal_initializer(), bottom_layer=None,
                 optimizer=None, output_layer='MSE'):

        self.learning_rate = learning_rate
        self.lam = 0
        self.output_layer = output_layer
        if optimizer is None:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        if bottom_layer is None:
            self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
            var_in = self.x
            trainable_vars = []
        else:
            self.x = bottom_layer.x
            var_in = bottom_layer.var_out
            trainable_vars = bottom_layer.trainable_vars
            self.bottom_layer = bottom_layer

        self.h = tf.placeholder(tf.float32, shape=(dim_hidden, dim_out))

        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))

        U, b_hidden, net, phi, W, b_out, y =\
            crossprop_layer(name, 'crossprop_layer', var_in, dim_in, dim_hidden, dim_out, gate.gate_fun, initializer)
        if self.output_layer == 'CE':
            self.pred = tf.nn.softmax(y)
            ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=self.target)
            self.loss = tf.reduce_mean(ce_loss)
            self.total_loss = tf.reduce_sum(ce_loss)
            correct_prediction = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.pred, 1))
            self.correct_labels = tf.reduce_sum(tf.cast(correct_prediction, "float"))
            delta = tf.subtract(self.pred, self.target)
        elif self.output_layer == 'MSE':
            se_loss = 0.5 * tf.squared_difference(y, self.target)
            self.loss = tf.reduce_mean(se_loss)
            self.total_loss = tf.reduce_sum(se_loss)
            delta = y - self.target
            self.correct_labels = tf.constant(0)
        else:
            assert False

        trainable_vars.extend([W, b_out])

        h_decay = tf.subtract(1.0, tf.scalar_mul(learning_rate, tf.pow(phi, 2)))
        h_decay = tf.reshape(tf.tile(h_decay, [1, tf.shape(self.h)[1]]), [-1, tf.shape(self.h)[1], tf.shape(self.h)[0]])
        h_decay = tf.transpose(h_decay, [0, 2, 1])
        self.h_decay = tf.reduce_sum(h_decay, axis=0)

        h_delta = tf.reshape(tf.tile(delta, [1, tf.shape(self.h)[0]]), [-1, tf.shape(self.h)[0], tf.shape(self.h)[1]])
        self.h_delta = tf.reduce_sum(h_delta, axis=0)

        new_grads = []
        phi_phi_grad = tf.multiply(phi, gate.gate_fun_gradient(phi, net))
        weight = tf.transpose(tf.matmul(self.h, tf.transpose(delta)))
        phi_phi_grad = tf.multiply(phi_phi_grad, weight)
        new_u_grad = tf.matmul(tf.transpose(var_in), phi_phi_grad)
        new_u_grad = tf.scalar_mul(1.0 / tf.cast(tf.shape(var_in)[0], tf.float32), new_u_grad)

        phi_error = tf.matmul(delta, tf.transpose(W))
        net_error = tf.multiply(phi_error, gate.gate_fun_gradient(phi, net))
        bp_u_grad = tf.matmul(tf.transpose(var_in), net_error)
        bp_u_grad = tf.scalar_mul(1.0 / tf.cast(tf.shape(var_in)[0], tf.float32), bp_u_grad)

        new_u_grad = (1 - self.lam) * new_u_grad + self.lam * bp_u_grad
        new_grads.append(new_u_grad)

        new_b_hidden_grad = tf.reduce_mean(phi_phi_grad, axis=0)
        bp_b_hidden_grad = tf.reduce_mean(net_error, axis=0)
        new_b_hidden_grad = (1 - self.lam) * new_b_hidden_grad + self.lam * bp_b_hidden_grad
        new_grads.append(new_b_hidden_grad)

        old_grads = optimizer.compute_gradients(self.loss, var_list=[U, b_hidden])
        for i, (grad, var) in enumerate(old_grads):
            old_grads[i] = (new_grads[i], var)
        other_grads = optimizer.compute_gradients(self.loss, var_list=trainable_vars)

        self.all_gradients = old_grads + other_grads
        self.train_op = optimizer.apply_gradients(self.all_gradients)
        self.h_var = np.zeros((dim_hidden, dim_out))
        self.variables = [W, b_out, U, b_hidden]
        self.y = y

    def get_assign_ops(self, src_network):
        assign_ops = []
        for dst_var, src_var in zip(self.variables, src_network.variables):
            assign_ops.append(dst_var.assign(src_var))
        return assign_ops

    def predict(self, sess, x):
        y = sess.run(self.y, feed_dict={self.x: x})
        return y

    def learn(self, sess, train_x, train_y):
        _, h_decay_var, h_delta_var = \
            sess.run([self.train_op, self.h_decay, self.h_delta],
                     feed_dict={
                         self.x: train_x,
                         self.target: train_y,
                         self.h: self.h_var
                     })
        batch_size = float(train_x.shape[0])
        self.h_var = np.multiply(h_decay_var / batch_size, self.h_var) - self.learning_rate * h_delta_var / batch_size
