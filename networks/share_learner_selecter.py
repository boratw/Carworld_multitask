
import numpy as np
import tensorflow as tf
from .mlp import MLP

class DQNSelecter:
    def __init__(self, pair_len, embed_len, name="", hidden_len = 64, learn_lr=0.001, lstm_sizes = [64, 64, 64]):
        with tf.variable_scope("DQNSelecter" + name): 
            self.input_pair = tf.placeholder(tf.float32, [None, pair_len], name="input_pair")
            self.input_target = tf.placeholder(tf.float32, [1, embed_len], name="input_target")
            self.input_learning_rate = tf.placeholder(tf.float32, [], name="input_learning_rate")

            w1 = tf.get_variable("w1", shape=[pair_len, hidden_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (pair_len + hidden_len)), np.sqrt(1.0 / (pair_len + hidden_len)), dtype=tf.float32),
                trainable=True)
            b1 = tf.get_variable("b1", shape=[hidden_len], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
            fc1 = tf.matmul(self.input_pair, w1) + b1
            fc1 = tf.nn.leaky_relu(fc1)
            lstm_input = tf.reshape(fc1, [1, -1, hidden_len])

            lstms = [tf.nn.rnn_cell.BasicLSTMCell(size) for size in lstm_sizes] 
            drops = [tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.9) for lstm in lstms]
            cell = tf.nn.rnn_cell.MultiRNNCell(drops)

            self.zero_state = cell.zero_state(1, tf.float32)
            state_variables = []
            for state_c, state_h in self.zero_state:
                state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(tf.Variable(state_c, trainable=False), tf.Variable(state_h, trainable=False)))
            state_variables = tuple(state_variables)

            self.lstm_output, new_states = tf.nn.dynamic_rnn(cell, lstm_input, initial_state=state_variables)
            self.update_op = []
            for state_variable, new_state in zip(state_variables, new_states):
                self.update_op.extend([state_variable[0].assign(new_state[0]), state_variable[1].assign(new_state[1])])
            self.reset_op = []
            for state_variable, new_state in zip(state_variables, self.zero_state):
                self.reset_op.extend([state_variable[0].assign(new_state[0]), state_variable[1].assign(new_state[1])])


            w2 = tf.get_variable("w2", shape=[hidden_len, embed_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (hidden_len + embed_len)), np.sqrt(1.0 / (hidden_len + embed_len)), dtype=tf.float32),
                trainable=True)
            b2 = tf.get_variable("b2", shape=[embed_len], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            self.output = tf.matmul(self.lstm_output[-1], w2) + b2

            self.loss = tf.reduce_mean((self.output - self.input_target) ** 2) 
            self.train = tf.train.GradientDescentOptimizer(self.input_learning_rate).minimize(loss = self.loss)



    def get_output(self, input_pair):
        sess = tf.get_default_session()
        output, _ = sess.run([self.output, self.update_op], {self.input_pair : input_pair})
        return output

    def optimize(self, input_pair, input_target, input_learning_rate):
        input_list = {self.input_pair : input_pair, self.input_target : input_target, self.input_learning_rate : input_learning_rate}

        sess = tf.get_default_session()
        l, _, _= sess.run([self.loss, self.train, self.update_op], input_list)
        return np.mean(l)

    def reset(self):
        sess = tf.get_default_session()
        sess.run(self.reset_op)
