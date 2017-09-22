import tensorflow as tf
import numpy as np

class ValueNetwork:
	def __init__(self, state_dim, lr, save_path='model/value_network'):
		self.state_dim = state_dim
		self.lr = lr
		self.save_path = save_path

		self._build_graph()

		self.sess = tf.Session(graph=self.g)
		self.sess.run(self.init)

		try:
			self.saver.restore(self.sess, self.save_path)
			print ('Saved variables restored from checkpoint: {}'.format(save_path))
		except:
			pass

	def _build_graph(self):
		self.g = tf.Graph()
		with self.g.as_default():
			# Placeholders
			self.states_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
			self.targets_ph = tf.placeholder(tf.float32, shape=(None))

			# Hidden layer sizes
			h1_size = self.state_dim*10
			h3_size = 5
			h2_size = int(np.sqrt(h1_size*h3_size))

			# Network weights
			'''one layer'''
			# self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, 1)))
			# self.pred_value = tf.matmul(self.states_ph, self.w1)

			'''hidden layers'''
			self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			self.w2 = tf.Variable(tf.truncated_normal(shape=(h1_size, h2_size)))
			self.w3 = tf.Variable(tf.truncated_normal(shape=(h2_size, h3_size)))
			self.w4 = tf.Variable(tf.truncated_normal(shape=(h3_size, 1)))

			'''one hidden layer'''
			# self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			# self.w2 = tf.Variable(tf.truncated_normal(shape=(h1_size, 1)))
			# a1 = tf.matmul(self.states_ph, self.w1)
			# a1 = tf.nn.relu(a1)
			# self.pred_value = tf.matmul(a1, self.w2)

			# Graph operations
			a1 = tf.matmul(self.states_ph, self.w1)
			a2 = tf.matmul(a1, self.w2)
			a3 = tf.matmul(a2, self.w3)
			self.pred_value = tf.matmul(a3, self.w4)
			

			# Optimisation-related
			self.loss = tf.losses.absolute_difference(self.targets_ph, self.pred_value)
			self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

			# Init operation
			self.init = tf.global_variables_initializer()

			# Saver
			self.saver = tf.train.Saver()			

	def get_value(self, state):
		# print (self.states_ph.shape)
		# print (state.shape)
		# state = np.zeros_like(state)
		# state = np.zeros((200, 3))
		value = self.sess.run(self.pred_value, feed_dict={self.states_ph: state})
		return value

	def update(self, states, target_values):
		feed_dict = {self.states_ph: states, self.targets_ph: target_values}
		self.sess.run(self.opt, feed_dict=feed_dict)

	def print_for_debug(self):
		print (self.sess.run(self.w1))

	def save_model(self):
		save_name = self.saver.save(self.sess, self.save_path)
		print("Value network checkpoint saved in file: %s" % save_name)
