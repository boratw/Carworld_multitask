
import numpy as np
import tensorflow as tf
from .mlp import MLP

class DQNSharedLearner:
    def __init__(self, state_len, action_len, name="", hidden_len=128, embed_len=[32, 32], qvalue_lr=0.001, state_lr = 0.001, gamma=0.96):
        self.gamma = gamma
        with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable("w1", shape=[state_len, hidden_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (state_len + hidden_len)), np.sqrt(1.0 / (state_len + hidden_len)), dtype=tf.float32),
                trainable=True)
            #b1 = tf.get_variable("b1", shape=[hidden_len], dtype=tf.float32, 
            #    initializer=tf.zeros_initializer(dtype=tf.float32),
            #    trainable=True)
            w2 = tf.get_variable("w2", shape=[hidden_len, embed_len[0]], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (hidden_len + embed_len[0])), np.sqrt(1.0 / (hidden_len + embed_len[0])), dtype=tf.float32),
                trainable=True)
            #b2 = tf.get_variable("b2", shape=[embed_len[0]], dtype=tf.float32, 
            #    initializer=tf.zeros_initializer(dtype=tf.float32),
            #    trainable=True)
            w3 = tf.get_variable("w3", shape=[embed_len[1], action_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (embed_len[1] + action_len)), np.sqrt(1.0 / (embed_len[1] + action_len)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[action_len], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

        with tf.variable_scope("DQNLearner" + name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_survive = tf.placeholder(tf.float32, [None, 1], name="input_survive")
            self.input_value = tf.placeholder(tf.float32, [None, 1], name="input_value")
            
            self.embedding = tf.get_variable("embedding", shape=[embed_len[0], embed_len[1]], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (embed_len[0] + embed_len[1])), np.sqrt(1.0 / (embed_len[0] + embed_len[1])), dtype=tf.float32),
                trainable=True)
            self.embedding_flat = tf.reshape(self.embedding, [1, -1])

            fc1 = tf.matmul(self.input_state, w1)# + b1
            fc1 = tf.nn.tanh(tf.nn.dropout(fc1, 0.9))
            fc2 = tf.matmul(fc1, w2)# + b2
            fc2 = tf.nn.tanh(tf.nn.dropout(fc2, 0.9))
            fc2_embedded = tf.matmul(fc2, self.embedding)
            fc2_embedded = tf.nn.leaky_relu(tf.nn.dropout(fc2_embedded, 0.9))
            self.output_qvalue = tf.matmul(fc2_embedded, w3) + b3

            fc1 = tf.matmul(self.input_next_state, w1)# + b1
            fc1 = tf.nn.tanh(tf.nn.dropout(fc1, 0.9))
            fc2 = tf.matmul(fc1, w2)# + b2
            fc2 = tf.nn.tanh(tf.nn.dropout(fc2, 0.9))
            fc2_embedded = tf.matmul(fc2, self.embedding)
            fc2_embedded = tf.nn.leaky_relu(tf.nn.dropout(fc2_embedded, 0.9))
            self.output_next_qvalue = tf.matmul(fc2_embedded, w3) + b3


            dest_qvalue = tf.stop_gradient(self.input_reward + self.output_next_qvalue * self.input_survive * self.gamma)
            final_qvalue = tf.maximum(dest_qvalue, self.input_value)
            self.target_qvalue = tf.reduce_sum(self.input_action * self.output_qvalue, axis=1, keepdims=True)

            self.qvalue_loss = tf.reduce_mean((self.target_qvalue - final_qvalue) ** 2) 
            self.qvalue_train = tf.train.AdamOptimizer(qvalue_lr).minimize(loss = self.qvalue_loss, var_list = [self.embedding])
            self.qvalue_global_train = tf.train.AdamOptimizer(qvalue_lr / 10.).minimize(loss = self.qvalue_loss, var_list = [w1, w2, w3, b3, self.embedding])


    def get_action(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.output_qvalue, {self.input_state : np.array([input_state])})
        return np.argmax(output[0])

    def get_action_with_qvalue(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.output_qvalue, {self.input_state : np.array([input_state])})
        return np.argmax(output[0]), output[0]

    def get_action_random(self, input_state, random_scale):
        sess = tf.get_default_session()
        output = sess.run(self.output_qvalue, {self.input_state : np.array([input_state])})
        output = output[0]
        rnd = np.random.normal(0.0, random_scale, size=output.shape)
        output += rnd
        return np.argmax(output)

    def get_action_random_with_qvalue(self, input_state, random_scale):
        sess = tf.get_default_session()
        output = sess.run(self.output_qvalue, {self.input_state : np.array([input_state])})
        output = output[0]
        rnd = np.random.normal(0.0, random_scale, size=output.shape)
        output_r = output + rnd
        return np.argmax(output_r), output

    def optimize(self, input_state, input_next_state, input_action, input_reward, input_survive, input_value):
        input_list = {self.input_state : np.array([input_state]), self.input_next_state : np.array([input_next_state]), 
            self.input_action : np.array([input_action]), self.input_reward : np.array([input_reward]), self.input_survive : np.array([input_survive]),
            self.input_value : np.array([input_value])}

        sess = tf.get_default_session()
        q, ql, _= sess.run([self.output_qvalue, self.qvalue_loss, self.qvalue_train], input_list)
        q = q.reshape(-1)

        return np.mean(q), np.std(q), ql

    def optimize_batch(self, input_state, input_next_state, input_action, input_reward, input_survive, input_value):
        input_list = {self.input_state : np.array(input_state), self.input_next_state : np.array(input_next_state), 
            self.input_action : np.array(input_action), self.input_reward : np.array(input_reward), self.input_survive : np.array(input_survive),
            self.input_value : np.array(input_value)}

        sess = tf.get_default_session()
        q, ql, _= sess.run([self.output_qvalue, self.qvalue_loss, self.qvalue_train], input_list)
        q = q.reshape(-1)

        return np.mean(q), np.std(q), ql

    def optimize_global_batch(self, input_state, input_next_state, input_action, input_reward, input_survive, input_value):
        input_list = {self.input_state : np.array(input_state), self.input_next_state : np.array(input_next_state), 
            self.input_action : np.array(input_action), self.input_reward : np.array(input_reward), self.input_survive : np.array(input_survive),
            self.input_value : np.array(input_value)}

        sess = tf.get_default_session()
        q, ql, _= sess.run([self.output_qvalue, self.qvalue_loss, self.qvalue_global_train], input_list)
        q = q.reshape(-1)

        return np.mean(q), np.std(q), ql

    def get_embedding(self):
        sess = tf.get_default_session()
        res = sess.run(self.embedding_flat)
        return res
