import tensorflow as tf
import numpy as np
import random

class PolicyLearner:
	def __init__(self, n_actions, state_shape, discount=0.8, alpha=0.01, beta=0.01, lambda_w=0.5, lambda_theta=0.5): # beta isn't currently being used
		self.n_actions = n_actions
		self.state_shape = state_shape
		self.discount = discount
		self.alpha = alpha
		self.beta = beta
		self.lambda_w = lambda_w
		self.lambda_theta = lambda_theta

		tf.reset_default_graph()
		tf.set_random_seed(2)
		self.state_tensor, self.value_tensor, self.chosen_action_index, self.already_chosen, self.log_chosen_action_tensor, self.w_opt, self.theta_opt, self.theta, self.w = self._build_model()

		self.w_grads_and_vars = self._get_grads_and_vars(self.w_opt, self.value_tensor, 'value')
		self.theta_grads_and_vars = self._get_grads_and_vars(self.theta_opt, self.log_chosen_action_tensor, 'policy')
		self.w_gradients = [gv[0] for gv in self.w_grads_and_vars]
		self.theta_gradients = [gv[0] for gv in self.theta_grads_and_vars]

		self.w_e_trace = self._get_e_trace(self.w_gradients)
		self.theta_e_trace = self._get_e_trace(self.theta_gradients)

		self.I = 1

		# putting this up here so we don't keep adding to the graph with evey loop. it was previously causing the slow down
		self.w_grad_placeholder = [(tf.placeholder("float", shape=gv[0].get_shape()), gv[1]) for gv in self.w_grads_and_vars]
		self.theta_grad_placeholder = [(tf.placeholder("float", shape=gv[0].get_shape()), gv[1]) for gv in self.theta_grads_and_vars]
		self.apply_w_placeholder_op = self.w_opt.apply_gradients(self.w_grad_placeholder)
		self.apply_theta_placeholder_op = self.theta_opt.apply_gradients(self.theta_grad_placeholder)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		state_tensor = tf.placeholder(tf.float32, shape=(self.state_shape))
		w = tf.Variable(tf.truncated_normal(shape=(self.n_states, 1)), name='value_weight')
		value_tensor = tf.matmul(state_tensor, w)

		theta = tf.Variable(tf.truncated_normal(shape=(self.n_states, self.n_actions)), name='policy_weight')
		action_logits = tf.matmul(state_tensor, theta)
		action_probabilities = tf.nn.softmax(action_logits) # doing this softmax here makes the gradient depend on the entire weight
		chosen_action_index = tf.multinomial(tf.log(action_probabilities), num_samples=1) # picking according to probability

		# allowing us to get the gradient of the log probability of the CHOSEN action when it was chosen in the previous timestep and returned to the main script
		already_chosen = tf.placeholder(tf.int32, shape=[])
		chosen_action_prob_tensor = tf.gather(action_probabilities, already_chosen, axis=1)

		log_chosen_action_tensor = tf.log(chosen_action_prob_tensor)
		w_opt = tf.train.AdamOptimizer(self.alpha)
		theta_opt = tf.train.AdamOptimizer(self.beta)

		return state_tensor, value_tensor, chosen_action_index, already_chosen, log_chosen_action_tensor, w_opt, theta_opt, theta, w

	def _get_grads_and_vars(self, opt, target_tensor, variable_identifier):
		variables = [var for var in tf.global_variables() if variable_identifier in var.op.name] # tf.get_variable() only works if the variable was created using tf.get_variable()
		grads_and_vars = opt.compute_gradients(target_tensor, variables)
		return grads_and_vars

	def _get_e_trace(self, gradient):
		e_trace = []
		for gradient in gradient:
			e = np.zeros(gradient.get_shape())
			e_trace.append(e)
		return e_trace

	def _update_e_trace(self, evaluated_gradients, e_trace, lamb):
		for i in range(len(e_trace)):
			e_trace[i] = lamb*e_trace[i] + self.I*evaluated_gradients[i]
			assert(e_trace[i].shape == evaluated_gradients[i].shape)
		return e_trace

	def _get_value(self, state):
		return self.sess.run(self.value_tensor, feed_dict={self.state_tensor: state})

	def get_action(self, state):
		action = self.sess.run(self.chosen_action_index, feed_dict={self.state_tensor: state})
		return np.asscalar(action)

	def reset_e_trace(self):
		self.w_e_trace = [0*e for e in self.w_e_trace]
		self.theta_e_trace = [0*e for e in self.theta_e_trace]
		self.I = 1

	def print_for_debug(self):
		print ('Theta:')
		print (self.sess.run(self.theta))
		print ('w:')
		print (self.sess.run(self.w))

	def learn(self, state, action, next_state, reward):
		target = reward + self.discount*self._get_value(next_state)
		old_value = self._get_value(state)
		delta = target - old_value

		w_gradients_evaluated = self.sess.run(self.w_gradients, feed_dict={self.state_tensor: state})
		self.w_e_trace = self._update_e_trace(w_gradients_evaluated, self.w_e_trace, self.lambda_w)
		theta_gradients_evaluated = self.sess.run(self.theta_gradients, feed_dict={self.state_tensor: state, self.already_chosen: action})
		self.theta_e_trace = self._update_e_trace(theta_gradients_evaluated, self.theta_e_trace, self.lambda_theta)

		w_change = [-delta * e for e in self.w_e_trace]
		theta_change = [-delta * e for e in self.theta_e_trace]

		# For debugging purposes. Initially had problems with numbers getting too large.
		if abs(delta)>1000:
			print ('Warning! Delta getting very big. Delta = {}'.format(delta))

		# print (state)
		# print (theta_gradients_evaluated)

		# APPLY GRADIENT UPDATE. Eligibility trace (e_trace) is essentially a modified gradient. change is the change to be applied to the weights.
		# To alter the gradients before applying them, we have to do some session running and dictionary feeding
		feed_dict_w = {}
		for i in range(len(self.w_grad_placeholder)):
			feed_dict_w[self.w_grad_placeholder[i][0]] = w_change[i]
		feed_dict_theta = {}
		for i in range(len(self.theta_grad_placeholder)):
			feed_dict_theta[self.theta_grad_placeholder[i][0]] = theta_change[i]

		self.sess.run(self.apply_w_placeholder_op, feed_dict=feed_dict_w)
		self.sess.run(self.apply_theta_placeholder_op, feed_dict=feed_dict_theta)

		self.I = self.I*self.discount