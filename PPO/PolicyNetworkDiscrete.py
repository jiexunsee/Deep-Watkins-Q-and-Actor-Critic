import tensorflow as tf
import numpy as np

class PolicyNetworkDiscrete:
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
			self.saver.restore(self.sess, self.save_path)
			print ('Saved variables restored from checkpoint: {}'.format(save_path))
		except:
			pass

	def _build_graph(self):
		self.g = tf.Graph()
		with self.g.as_default():
			# Placeholders
			self.states_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
			self.actions_ph = tf.placeholder(tf.int32, shape=(None,))
			self.advs_ph = tf.placeholder(tf.float32, shape=(None,))
			self.old_probabilities_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))

			# Hidden layer sizes
			# h1_size = self.state_dim*10
			# h3_size = self.action_dim*10
			# h2_size = np.sqrt(h1_size*h3_size)

			# Network weights
			'''one layer'''
			# self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, self.action_dim)))
			# a2 = tf.matmul(self.states_ph, self.w1)

			'''hidden layers'''
			# self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			# self.w2 = tf.Variable(tf.truncated_normal(shape=(h1_size, h2_size)))
			# self.w3 = tf.Variable(tf.truncated_normal(shape=(h2_size, h3_size)))
			# self.w4 = tf.Variable(tf.truncated_normal(shape=(h3_size, self.action_dim)))

			'''one hidden layer'''
			h1_size = 200
			self.w1 = tf.Variable(tf.truncated_normal(shape=(self.state_dim, h1_size)))
			self.w2 = tf.Variable(tf.truncated_normal(shape=(h1_size, self.action_dim)))
			a1 = tf.matmul(self.states_ph, self.w1)
			a1 = tf.nn.relu(a1)
			a2 = tf.matmul(a1, self.w2)

			# Graph operations
			# a1 = tf.matmul(self.states_ph, self.w1)
			# a2 = tf.matmul(a1, self.w2)
			# a3 = tf.matmul(a2, self.w3)

			self.probabilities = tf.nn.softmax(a2) # doing this softmax here makes the gradient depend on the entire weight
			self.probabilities = tf.clip_by_value(self.probabilities,1e-10,1.0-1e-10)
			self.sampled_act = tf.multinomial(tf.log(self.probabilities), num_samples=1) # picking according to probability

			# Optimisation-related
			self.logp = self.find_log_prob(self.actions_ph, self.probabilities)
			self.logp_old = self.find_log_prob(self.actions_ph, self.old_probabilities_ph)
			self.ratio = tf.exp(self.logp - self.logp_old)
			clipped_ratio = tf.clip_by_value(self.ratio, 1-self.e, 1+self.e)
			objective = tf.minimum(self.ratio*self.advs_ph, clipped_ratio*self.advs_ph)
			self.loss = tf.reduce_mean(-objective)
			self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
			# Saver
			self.saver = tf.train.Saver()

			# Init operation
			self.init = tf.global_variables_initializer()


	def find_log_prob(self, actions, probabilities):
		order = tf.range(0, tf.shape(actions)[0])
		self.indices = tf.stack([order, actions], axis=1)
		logp = tf.log(tf.gather_nd(probabilities, self.indices))
		return logp

	def get_action(self, state):
		action = self.sess.run(self.sampled_act, feed_dict={self.states_ph: state})
		return action

	def print_for_debug(self):
		w1 = self.sess.run(self.w1)
		print (w1)

	def update(self, states, actions, advantages):
		feed_dict = {self.states_ph: states, self.actions_ph: actions, self.advs_ph: advantages}
		old_probabilities = self.sess.run(self.probabilities, feed_dict)
		feed_dict[self.old_probabilities_ph] = old_probabilities

		_, r, lp, lpo, prob, o = self.sess.run([self.opt, self.ratio, self.logp, self.logp_old, self.probabilities, self.indices], feed_dict=feed_dict)
		# r, lp, lpo, prob, o = self.sess.run([self.ratio, self.logp, self.logp_old, self.probabilities, self.indices], feed_dict=feed_dict)
		# print (len(states))
		# print (actions.shape)
		# print (lpo.shape)
		# print (r) # R IS ALL ONES!
		# print (prob.shape)
		# print (o.shape)

	def save_model(self):
		save_name = self.saver.save(self.sess, self.save_path)
		print("Policy network checkpoint saved in file: %s" % save_name)