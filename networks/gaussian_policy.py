
import numpy as np
import tensorflow as tf
import math

EPS = 1e-6

class GaussianPolicy:
    def __init__(self, name, state_len, action_len, hidden_sizes=64, hidden_nonlinearity=tf.nn.relu, output_nonlinearity=tf.nn.tanh, 
        reuse=False, reg=0.001, input_tensor=None, action_limit = 1.0):

        with tf.variable_scope(name, reuse=reuse):
                
            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, state_len])
            else:
                self.layer_input = input_tensor
            

            w1 = tf.get_variable("w1", shape=[state_len, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / (state_len + hidden_sizes)), math.sqrt(1.0 / (state_len + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b1 = tf.get_variable("b1", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc1 = tf.matmul(self.layer_input, w1) + b1
            if hidden_nonlinearity is not None:
                fc1 = hidden_nonlinearity(fc1)


            w2 = tf.get_variable("w2", shape=[hidden_sizes, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / (hidden_sizes + hidden_sizes)), math.sqrt(1.0 / (hidden_sizes + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b2 = tf.get_variable("b2", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc2 = tf.matmul(fc1, w2) + b2
            if hidden_nonlinearity is not None:
                fc2 = hidden_nonlinearity(fc2)


            w3 = tf.get_variable("w3", shape=[hidden_sizes, action_len * 2], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / (hidden_sizes + action_len)), math.sqrt(1.0 / (hidden_sizes + action_len)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[action_len * 2], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc3 = tf.matmul(fc2, w3) + b3

            if output_nonlinearity is not None:
                fc3 = output_nonlinearity(fc3)

            self.reg = reg

            self.mu, self.logsig = tf.split(fc3, [action_len, action_len], 1)
            self.mu = self.mu * action_limit
            self.std = tf.exp(self.logsig)
            #self.logsig = tf.clip_by_value(self.logsig, -20, 2)
            #self.logsig_walk = tf.clip_by_value(self.logsig, -5, -1)


            self.dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=self.std)
            self.sampled = self.dist.sample()
            self.x = tf.clip_by_value(self.sampled, -action_limit, action_limit)
            self.stopped_x = tf.stop_gradient(self.x)

            self.dist_walk = tf.contrib.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.logsig + 1))
            self.walk = tf.clip_by_value(self.dist_walk.sample(), -action_limit, action_limit)


            #self.log_pi = self.dist.log_prob(self.x)# - self.squash_correction(self.squashed_x)
            self.log_pi = self.dist.log_prob(self.x)

            self.regularization_loss =  0.0001 * tf.reduce_sum( tf.reduce_mean((self.mu / action_limit) ** 2) + tf.reduce_mean(self.logsig ** 2))

            self.trainable_params = [w1, b1, w2, b2, w3, b3]

    #def log_prob(self, action):
    #    z = (action - self.mu)
    #    return self.logsig + 0.5 * z ** 2

 
    def squash_correction(self, actions):
        return tf.reduce_sum(tf.log(1 - actions ** 2 + EPS), axis=1)

    def build_assign(self, source):
        return [ tf.assign(target, source) for target, source in zip(self.trainable_params, source.trainable_params)]
