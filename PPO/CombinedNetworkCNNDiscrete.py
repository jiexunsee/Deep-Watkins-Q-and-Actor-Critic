import tensorflow as tensorflow
import numpy as np

# Following architecture from here https://github.com/andreimuntean/A3C/blob/master/a3c.py
def _convolutional_layer(x, shape, stride, activation_fn):
    if len(shape) != 4:
        raise ValueError('Shape "{}" is invalid. Must have length 4.'.format(shape))
    num_input_params = shape[0] * shape[1] * shape[2]
    num_output_params = shape[0] * shape[1] * shape[3]
    maxval = math.sqrt(6 / (num_input_params + num_output_params))
    W = tf.Variable(tf.random_uniform(shape, -maxval, maxval), name='Weights')
    b = tf.Variable(tf.zeros(shape=[shape[3]]), name='Bias')
    conv = tf.nn.conv2d(x, W, [1, stride, stride, 1], 'VALID')

    return activation_fn(tf.nn.bias_add(conv, b))

def _fully_connected_layer(x, shape, activation_fn):
    if len(shape) != 2:
        raise ValueError('Shape "{}" is invalid. Must have length 2.'.format(shape))
    maxval = math.sqrt(6 / (shape[0] + shape[1]))
    W = tf.Variable(tf.random_uniform(shape, -maxval, maxval), name='Weights')
    b = tf.Variable(tf.constant(0, tf.float32, [shape[1]]), name='Bias')

    return activation_fn(tf.matmul(x, W) + b)

class CombinedNetworkCNNDiscrete:
	def __init__(action_dim, state_dim, lr):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = lr

		self._build_graph()

		self.sess = tf.Session(graph=self.g)
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		self.g = tf.Graph()
		with self.g.as_default():
			# Placeholders
			self.states_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
			self.actions_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
			self.advs_ph = tf.placeholder(tf.float32, shape=(None,))
			self.old_means_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
			self.old_log_vars_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))

			# Hidden layer sizes
			h1_size = self.state_dim*10
			h3_size = self.action_dim*10
			h2_size = np.sqrt(h1_size*h3_size)

			# Network weights
			self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			self.w2 = tf.Variable(tf.truncated_normal(shape=(self.h1_size, self.h2_size)))
			self.w3 = tf.Variable(tf.truncated_normal(shape=(self.h2_size, self.h3_size)))
			self.w4 = tf.Variable(tf.truncated_normal(shape=(self.h3_size, self.action_dim)))
			self.log_vars = tf.Variable(tf.truncated_normal(shape=(1, self.action_dim)))

			# Graph operations
			a1 = tf.matmul(self.states_ph, self.w1)
			a2 = tf.matmul(a1, self.w2)
			a3 = tf.matmul(a2, self.w3)
			self.pred_means = tf.matmul(a3, self.w4)
			self.sampled_act = (pred_means + tf.exp(self.log_vars/2.0) * tf.random_normal(shape=(self.action_dim,)))

			# Optimisation-related
			self.logp = self.find_log_prob(self.act_ph, self.means, self.log_vars)
			self.logp_old = self.find_log_prob(self.act_ph, self.old_means_ph, self.old_log_vars_ph)
			ratio = tf.exp(self.logp - self.logp_old)
			clipped_ratio = tf.clip_by_value(ratio, 1-self.e, 1+self.e)
			objective = tf.minimum(ratio*self.advs_ph, clipped_ratio*self.advs_ph)
			self.loss = tf.reduce_mean(objective)
			self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def find_log_prob(self, action, means, log_vars):
		logp = -0.5 * tf.reduce_sum(log_vars)
		logp += -0.5 * tf.reduce_sum(tf.square(action - means) / tf.exp(log_vars), axis=1)
		return logp

	def get_action(self, state):
		action = self.sess.run(self.sampled_act, feed_dict={self.states_ph: state})
		return action

	def update(self, states, actions, advantages):
		feed_dict = {self.states_ph: states, self.actions_ph: actions, self.advs_ph: advantages}
		old_means, old_log_vars = self.sess.run([self.pred_means, self.log_vars], feed_dict)
		feed_dict[self.old_log_vars_ph] = old_log_vars_np
		feed_dict[self.old_means_ph] = old_means_np

		self.sess.run(self.opt, feed_dict=feed_dict)