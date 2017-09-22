import tensorflow as tf
import numpy as np

class PolicyNetwork:
	def __init__(self, action_dim, state_dim, lr, save_path='model/policy_network'):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = lr
		self.e = 0.2
		self.save_path = save_path

		self._build_graph()

		self.sess = tf.Session(graph=self.g)
		self.sess.run(self.init)

		try:
			self.saver.restore(self.sess, save_path)
			print ('Saved variables restored from checkpoint: {}'.format(save_path))
		except:
			pass

	def _build_graph(self):
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
			h2_size = int(np.sqrt(h1_size*h3_size))

			# Network weights
			'''one layer'''
			# self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, self.action_dim)))
			# self.pred_means = tf.matmul(self.states_ph, self.w1)

			'''hidden layers'''
			self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			self.w2 = tf.Variable(tf.truncated_normal(shape=(h1_size, h2_size)))
			self.w3 = tf.Variable(tf.truncated_normal(shape=(h2_size, h3_size)))
			self.w4 = tf.Variable(tf.truncated_normal(shape=(h3_size, self.action_dim)))

			'''one hidden layer'''
			# h1_size = 200
			# self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			# self.w2 = tf.Variable(tf.truncated_normal(shape=(h1_size, self.action_dim)))
			# a1 = tf.matmul(self.states_ph, self.w1)
			# a1 = tf.nn.relu(a1)
			# a2 = tf.matmul(a1, self.w2)


			# Network weights
			# self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			# self.w2 = tf.Variable(tf.truncated_normal(shape=(h1_size, h2_size)))
			# self.w3 = tf.Variable(tf.truncated_normal(shape=(h2_size, h3_size)))
			# self.w4 = tf.Variable(tf.truncated_normal(shape=(h3_size, self.action_dim)))
			self.log_vars = tf.Variable(tf.truncated_normal(shape=(1, self.action_dim), stddev=0.01))

			# Graph operations
			a1 = tf.matmul(self.states_ph, self.w1)
			a2 = tf.matmul(a1, self.w2)
			a3 = tf.matmul(a2, self.w3)
			self.pred_means = tf.matmul(a3, self.w4)
			self.sampled_act = (self.pred_means + tf.exp(self.log_vars/2.0) * tf.random_normal(shape=(self.action_dim,)))

			# Optimisation-related
			self.logp = self.find_log_prob(self.actions_ph, self.pred_means, self.log_vars)
			self.logp_old = self.find_log_prob(self.actions_ph, self.old_means_ph, self.old_log_vars_ph)
			ratio = tf.exp(self.logp - self.logp_old)
			clipped_ratio = tf.clip_by_value(ratio, 1-self.e, 1+self.e)
			objective = tf.minimum(ratio*self.advs_ph, clipped_ratio*self.advs_ph)
			self.loss = tf.reduce_mean(-objective)
			self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

			# Init operation
			self.init = tf.global_variables_initializer()

			# Saver
			self.saver = tf.train.Saver()

	def find_log_prob(self, action, means, log_vars):
		logp = -0.5 * tf.reduce_sum(log_vars)
		logp += -0.5 * tf.reduce_sum(tf.square(action - means) / tf.exp(log_vars), axis=1)
		return logp

	def get_action(self, state):
		action = self.sess.run(self.sampled_act, feed_dict={self.states_ph: state})
		return action

	def update(self, states, actions, advantages):
		feed_dict = {self.states_ph: states, self.actions_ph: actions, self.advs_ph: advantages}
		# print (states.shape)
		# print (actions.shape)
		# print (advantages.shape)
		old_means, old_log_vars = self.sess.run([self.pred_means, self.log_vars], feed_dict)
		feed_dict[self.old_log_vars_ph] = old_log_vars
		feed_dict[self.old_means_ph] = old_means

		_ = self.sess.run([self.opt], feed_dict=feed_dict)

	def print_for_debug(self):
		w1 = self.sess.run(self.w1)
		print (w1)

	def save_model(self):
		save_name = self.saver.save(self.sess, self.save_path)
		print("Policy network checkpoint saved in file: %s" % save_name)