import tensorflow as tf
import numpy as np
import random

from ValueNetwork import ValueNetwork
from PolicyNetwork import PolicyNetwork

class PPOLearner:
	def __init__(self, action_dim, state_dim, discount, lam, value_lr, policy_lr):
		self.discount = discount
		self.lam = lam

		tf.reset_default_graph()

		self.value_network = ValueNetwork(state_dim, value_lr)
		self.policy_network = PolicyNetwork(action_dim, state_dim, policy_lr)

		try:
			self.saver.restore(self.sess, save_path)
			print ('Saved variables restored from checkpoint: {}'.format(save_path))
		except:
			pass


	def collect_data(self, env, N, T):
		state_data, action_data, reward_data = [], [], []
		for i in range(N):
			states, actions, rewards = self.run_timesteps(env, T)
			state_data.append(states)
			action_data.append(action_data)
			reward_data.append(reward_data)

		return state_data, action_data, reward_data

	def run_timesteps(self, env, T):
		states, actions, rewards = [], [], [] # states will have one more entry than actions, rewards
		state = env.reset()
		state = np.reshape(state, (1, -1))
		states.append(state)
		for j in range(T):
			state = np.reshape(state, (1, -1))
			action = self.policy_network.get_action(state)
			next_state, reward, done, _ = env.step(action)

			states.append(next_state)
			actions.append(action)
			rewards.append(reward)

			if done:
				break

			state = next_state

		return states, actions, rewards			


	def compute_targets_and_advantages(self, state_data, action_data, reward_data):
		target_data = []
		advantage_data = []

		assert(len(state_data[0]) == len(action_data[0]+1)) # checking the trajectory data is correct format

		for i in range(len((state_data))):
			states = state_data[i]
			actions = action_data[i]
			rewards = reward_data[i]

			targets = []
			advantages = []

			for i in range(actions):
				target, advantage = self.compute_timestep_target_and_advantage(states[i:], rewards[i:])
				targets.append(target)
				advantages.append(advantage)

			target_data.append(targets)
			advantage_data.append(advantages)

		return target_data, advantage_data

	def compute_timestep_target_and_advantage(self, states, rewards):
		last_value = self.value_network.get_value(states[-1])
		rewards_term = [rewards[i] * (self.lam*self.discount)**(i) for i in range(len(rewards))]
		values_term = [states[i+1] * (1-self.lam)*self.discount * (self.lam*self.discount)**(i) for i in range(len(states)-1)]
		target = sum(rewards_term) + sum(values_term)
		advantage = target - last_value

		return target, advantage


	def train(self, env, iterations, N, T, epochs, M):
		for i in range(iterations):
			state_data, action_data, reward_data = self.collect_data(env, N, T)
			target_data, advantage_data = self.compute_targets_and_advantages(state_data, action_data, reward_data)

			for j in range(epochs):
				batch_generator = self.get_batches(M, state_data, action_data, target_data, advantage_data)
				for s, a, t, adv in batch_generator:
					self.value_network.update(s, t)
					self.policy_network.update(s, a, adv)


	def get_batches(self, M, state_data, action_data, target_data, advantage_data): # concatenate the states, actions... lists together into one whole list
		# need to remove last entry for each state
		flat_state_data = [s for s in states[:-1] for states in state_data]
		flat_action_data = [a for a in actions for actions in action_data]
		flat_target_data = [t for t in targets for targets in target_data]
		flat_advantage_data = [adv for adv in advantages for advantages in advantage_data]

		n_data = len(flat_state_data)
		randperm = random.sample(range(n_data), n_data)

		for i in range(0, n_data, M):
			end_i = min((i+1)*M, n_data-1)
			ii = randperm[i:end_i]
			yield flat_state_data[ii], flat_action_data[ii], flat_target_data[ii], flat_advantage_data[ii]

	def save_model(self):
		save_name = self.saver.save(self.sess, self.save_path)
		print("Model checkpoint saved in file: %s" % save_name)