import tensorflow as tf
import numpy as np
import random

from ValueNetwork import ValueNetwork
from PolicyNetwork import PolicyNetwork

class PPOLearner:
	def __init__(self, action_dim, state_dim, discount, lam, value_lr, policy_lr, value_path, policy_path):
		self.discount = discount
		self.lam = lam

		tf.reset_default_graph()

		self.value_network = ValueNetwork(state_dim, value_lr, value_path)
		self.policy_network = PolicyNetwork(action_dim, state_dim, policy_lr, policy_path)

	def collect_data(self, env, N, T):
		state_data, action_data, reward_data = [], [], []
		for i in range(N):
			states, actions, rewards = self.run_timesteps(env, T)
			state_data.append(states)
			action_data.append(actions)
			reward_data.append(rewards)

		return state_data, action_data, reward_data

	def run_timesteps(self, env, T):
		states, actions, rewards = [], [], [] # states will have one more entry than actions, rewards
		state = env.reset()
		state = np.squeeze(state)
		# state = np.reshape(state, (1, -1))
		states.append(state)
		for j in range(T):
			state = np.reshape(state, (1, -1))
			action = self.policy_network.get_action(state)
			next_state, reward, done, _ = env.step(action)
			env.render()
			
			# print (action)

			# reward = np.asscalar(reward)
			next_state = np.squeeze(next_state)

			if done:
				reward = -10

			states.append(next_state)
			actions.append(action)
			rewards.append(reward)

			if done:
				break

			state = next_state
		return states, actions, rewards			


	def compute_targets_and_advantages(self, state_data, reward_data):
		target_data = []
		advantage_data = []
		assert(len(state_data[0]) == len(reward_data[0])+1) # checking the trajectory data is correct format

		n = len(state_data)
		for i in range(n):
			states = state_data[i]
			rewards = reward_data[i]

			targets = []
			advantages = []

			states = np.vstack(states)
			# states = np.reshape(states, (-1, len(states[0])))
			values = self.value_network.get_value(states)
			for i in range(len(rewards)):
				target, advantage = self.compute_timestep_target_and_advantage(values[i:], rewards[i:])
				targets.append(target)
				advantages.append(advantage)

			target_data.append(targets)
			advantage_data.append(advantages)

		return target_data, advantage_data

	def compute_timestep_target_and_advantage(self, values, rewards):
		original_value = values[0]
		original_value = np.asscalar(original_value)
		rewards_term = [rewards[i] * (self.lam*self.discount)**(i) for i in range(len(rewards))]
		values_term = [values[i+1] * (1-self.lam)*self.discount * (self.lam*self.discount)**(i) for i in range(len(values)-1)]
		target = sum(rewards_term) + sum(values_term)
		try:
			target = np.asscalar(target)
		except:
			target = target
		advantage = target - original_value

		return target, advantage


	def train(self, env, iterations, N, T, epochs, M):
		for i in range(iterations):
			print ('\nCollecting data...')
			state_data, action_data, reward_data = self.collect_data(env, N, T)

			average_reward = np.sum(np.sum(reward_data))/len(reward_data)
			print ('Iteration {}: Average reward = {}'.format(i, average_reward))
			if np.isnan(average_reward):
				self.policy_network.print_for_debug()

			print ('Calculating targets and advantages...')
			target_data, advantage_data = self.compute_targets_and_advantages(state_data, reward_data)

			print ('Updating...')
			for j in range(epochs):
				batch_generator = self.get_batches(M, state_data, action_data, target_data, advantage_data)
				for s, a, t, adv in batch_generator:
					if (len(s) > 0):
						# print (s)
						s = np.vstack(s)
					else:
						print ('state')
						print (s)
						continue
					self.value_network.update(s, t)
					
					# if (len(a) > 1):
					# 	a = np.vstack(a)
					# else:
					# 	print (a)
					a = np.vstack(a)
					self.policy_network.update(s, a, adv)

			if i%100 == 0:
				self.policy_network.save_model()
				self.value_network.save_model()


	def get_batches(self, M, state_data, action_data, target_data, advantage_data): # concatenate the states, actions... lists together into one whole list
		# need to remove last entry for each state
		flat_state_data = np.concatenate(tuple(states[:-1] for states in state_data), axis=0)
		flat_action_data = np.concatenate(tuple(actions for actions in action_data), axis=0)
		flat_target_data = np.concatenate(tuple(targets for targets in target_data), axis=0)
		flat_advantage_data = np.concatenate(tuple(advantages for advantages in advantage_data), axis=0)

		# print (flat_state_data)
		# print ('--------------------------------')
		

		# flat_state_data = np.array([s for states in state_data for s in states[:-1]])
		# flat_action_data = np.array([a for actions in action_data for a in actions])
		# flat_target_data = np.array([t for targets in target_data for t in targets])
		# flat_advantage_data = np.array([adv for advantages in advantage_data for adv in advantages])

		n_data = len(flat_state_data)
		randperm = random.sample(range(n_data), n_data)

		for i in range(0, n_data, M):
			end_i = min((i+1)*M, n_data-1)
			ii = randperm[i:end_i]
			yield flat_state_data[ii], flat_action_data[ii], flat_target_data[ii], flat_advantage_data[ii]

	def save_model(self):
		save_name = self.saver.save(self.sess, self.save_path)
		print("Model checkpoint saved in file: %s" % save_name)