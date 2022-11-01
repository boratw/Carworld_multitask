
import numpy as np
import tensorflow as tf
from .mlp import MLP

class DQNNetwork:
    def __init__(self, name, state_len, action_len, hidden_len, reuse=False, input_tensor=None):
        with tf.variable_scope(name, reuse=reuse):
            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, state_len])
            else:
                self.layer_input = input_tensor

            w1 = tf.get_variable("w1", shape=[state_len, hidden_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (state_len + hidden_len)), np.sqrt(1.0 / (state_len + hidden_len)), dtype=tf.float32),
                trainable=True)
            b1 = tf.get_variable("b1", shape=[hidden_len], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
            fc1 = tf.matmul(self.layer_input, w1) + b1
            fc1 = tf.nn.leaky_relu(fc1)


            w2 = tf.get_variable("w2", shape=[hidden_len, hidden_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (hidden_len + hidden_len)), np.sqrt(1.0 / (hidden_len + hidden_len)), dtype=tf.float32),
                trainable=True)
            b2 = tf.get_variable("b2", shape=[hidden_len], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
            fc2 = tf.matmul(fc1, w2) + b2
            fc2 = tf.nn.leaky_relu(fc2)

            w3 = tf.get_variable("w3", shape=[hidden_len, action_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-np.sqrt(1.0 / (hidden_len + action_len)), np.sqrt(1.0 / (hidden_len + action_len)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[action_len], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
            self.output_qvalue = tf.matmul(fc2, w3) + b3
            self.max_qvalue = tf.reduce_max(self.output_qvalue, axis=1, keepdims=True)
            '''
            self.output_value, qvalue = tf.split(fc3, [1, action_len], 1)

            mean_qvalue = tf.reduce_mean(qvalue, axis=1, keepdims=True)
            mean_qvalues = tf.tile(mean_qvalue, [1, action_len])
            self.output_avalue = qvalue - mean_qvalues
            values = tf.tile(self.output_value, [1, action_len])
            self.output_qvalue = self.output_avalue + values
            self.max_qvalue = tf.reduce_max(self.output_qvalue, axis=1, keepdims=True)
            '''
            self.trainable_params = [w1, b1, w2, b2, w3, b3]


class DQNLearner:
    def __init__(self, state_len, action_len, name="", hidden_len=256, qvalue_lr=0.001, state_lr = 0.001, gamma=0.96):
        self.gamma = gamma
        self.zeroindex = action_len // 2

        with tf.variable_scope("DQNLearner" + name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_survive = tf.placeholder(tf.float32, [None, 1], name="input_survive")
            self.input_value = tf.placeholder(tf.float32, [None, 1], name="input_value")
            
            self.main_network = DQNNetwork("main", state_len, action_len, hidden_len, input_tensor=self.input_state)
            self.state_network = MLP("state", state_len, state_len, hidden_len, hidden_nonlinearity=tf.nn.leaky_relu,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.next_network = DQNNetwork("main", state_len, action_len, hidden_len, reuse=True, input_tensor=self.state_network.layer_output)

            self.target_network = DQNNetwork("target", state_len, action_len, hidden_len, input_tensor=self.input_state)

            dest_qvalue = tf.stop_gradient(self.input_reward + self.next_network.max_qvalue * self.input_survive * self.gamma)
            final_qvalue = tf.maximum(dest_qvalue, self.input_value)
            self.target_qvalue = tf.reduce_sum(self.input_action * self.target_network.output_qvalue, axis=1, keepdims=True)

            self.state_loss = tf.reduce_mean((self.input_next_state - self.state_network.layer_output) ** 2)
            self.state_train = tf.train.AdamOptimizer(state_lr).minimize(self.state_loss, var_list = self.state_network.trainable_params)

            self.qvalue_loss = tf.reduce_mean((self.target_qvalue - final_qvalue) ** 2) 
            self.qvalue_train = tf.train.AdamOptimizer(qvalue_lr).minimize(loss = self.qvalue_loss, var_list = self.target_network.trainable_params)

            self.network_assign1 = [ tf.assign(target, 0.9 * target + 0.1 * source) for target, source in zip(self.main_network.trainable_params, self.target_network.trainable_params)]
            self.network_assign2 = [ tf.assign(target, source) for target, source in zip(self.target_network.trainable_params, self.main_network.trainable_params)]


    def get_action(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.main_network.output_qvalue, {self.input_state : np.array([input_state])})
        return np.argmax(output[0])

    def get_action_with_qvalue(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.main_network.output_qvalue, {self.input_state : np.array([input_state])})
        return np.argmax(output[0]), output[0]

    def get_action_random(self, input_state, random_scale):
        sess = tf.get_default_session()
        output = sess.run(self.main_network.output_qvalue, {self.input_state : np.array([input_state])})
        output = output[0]
        rnd = np.random.normal(0.0, random_scale, size=output.shape)
        output += rnd
        return np.argmax(output)

    def get_action_random_with_qvalue(self, input_state, random_scale):
        sess = tf.get_default_session()
        output = sess.run(self.main_network.output_qvalue, {self.input_state : np.array([input_state])})
        output = output[0]
        rnd = np.random.normal(0.0, random_scale, size=output.shape)
        output_r = output + rnd
        return np.argmax(output_r), output

    def get_next_state(self, input_state, input_action):
        sess = tf.get_default_session()
        output = sess.run(self.state_network.layer_output, {self.input_state : np.array([input_state]), self.input_action : np.array([input_action])})
        return output[0]

    def optimize(self, input_state, input_next_state, input_action, input_reward, input_survive, input_value):
        input_list = {self.input_state : np.array([input_state]), self.input_next_state : np.array([input_next_state]), 
            self.input_action : np.array([input_action]), self.input_reward : np.array([input_reward]), self.input_survive : np.array([input_survive]),
            self.input_value : np.array([input_value])}

        sess = tf.get_default_session()
        q, ql, sl, _, _= sess.run([self.main_network.output_qvalue, self.qvalue_loss, self.state_loss, self.qvalue_train, self.state_train], input_list)
        q = q.reshape(-1)

        return np.mean(q), np.std(q), ql, sl

    def optimize_batch(self, input_state, input_next_state, input_action, input_reward, input_survive, input_value):
        input_list = {self.input_state : np.array(input_state), self.input_next_state : np.array(input_next_state), 
            self.input_action : np.array(input_action), self.input_reward : np.array(input_reward), self.input_survive : np.array(input_survive),
            self.input_value : np.array(input_value)}

        sess = tf.get_default_session()
        #v, a, q = sess.run([self.main_network.output_value, self.main_network.output_avalue,  self.main_network.output_qvalue], input_list)
        #print(v[0], a[0], q[0])
        q, ql, sl, _, _= sess.run([self.main_network.output_qvalue, self.qvalue_loss, self.state_loss, self.qvalue_train, self.state_train], input_list)
        q = q.reshape(-1)

        return np.mean(q), np.std(q), ql, sl

    def network_update(self):
        sess = tf.get_default_session()
        sess.run(self.network_assign1)
        sess.run(self.network_assign2)
