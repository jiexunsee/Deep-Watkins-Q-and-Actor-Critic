import tensorflow as tensorflow
import numpy as np

class ValueNetwork:
	def __init__(action_dim, state_dim, lr):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = lr

		self._build_graph()

		self.sess = tf.Session(graph=self.g)
		self.sess.run(tf.global_variables_initializer())

	def _build_graph(self):
		self.g = tf.Graph()
		with self.g.as_default():
			# Placeholders
			self.states_ph = tf.placeholder(tf.float, shape=(None, self.state_dim))
			self.targets_ph = tf.placeholder(tf.float, shape=(None))

			# Hidden layer sizes
			h1_size = self.state_dim*10
			h3_size = 5
			h2_size = np.sqrt(h1_size*h3_size)

			# Network weights
			self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			self.w2 = tf.Variable(tf.truncated_normal(shape=(h1_size, h2_size)))
			self.w3 = tf.Variable(tf.truncated_normal(shape=(h2_size, h3_size)))
			self.w4 = tf.Variable(tf.truncated_normal(shape=(h3_size, 1)))

			# Graph operations
			a1 = tf.matmul(self.states_ph, self.w1)
			a2 = tf.matmul(a1, self.w2)
			a3 = tf.matmul(a2, self.w3)
			self.pred_value = tf.matmul(a3, self.w4)

			# Optimisation-related
			self.loss = tf.losses.absolute_difference(self.targets_ph, self.pred_value)
			self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def get_value(self, state):
		value = self.sess.run(self.pred_value, feed_dict={self.states_ph: state})
		return value

	def update(self, states, target_values):
		feed_dict = {self.states_ph: states, self.targets_ph: target_values}
		self.sess.run(self.opt, feed_dict=feed_dict)
