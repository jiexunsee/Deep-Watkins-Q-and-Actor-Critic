import tensorflow as tf
import numpy as np
import random
# from tqdm import tqdm

from ValueNetwork import ValueNetwork
from PolicyNetworkDiscrete import PolicyNetworkDiscrete

def prepro(I):
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	I = I[35:195] # crop
	I = I[::2,::2,0] # downsample by factor of 2
	I[I == 144] = 0 # erase background (background type 1)
	I[I == 109] = 0 # erase background (background type 2)
	I[I != 0] = 1 # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel()

class PPOLearnerDiscrete:
	def __init__(self, action_dim, state_dim, discount, lam, value_lr, policy_lr):
		self.discount = discount
		self.lam = lam

		tf.reset_default_graph()

		self.value_network = ValueNetwork(state_dim, value_lr)
		self.policy_network = PolicyNetworkDiscrete(action_dim, state_dim, policy_lr)

	def collect_data(self, env, N, T, render):
		state_data, action_data, reward_data = [], [], []
		for i in range(N):
			states, actions, rewards = self.run_timesteps(env, T, render)
			state_data.append(states)
			action_data.append(actions)
			reward_data.append(rewards)

		return state_data, action_data, reward_data

	def run_timesteps(self, env, T, render):
		states, actions, rewards = [], [], [] # states will have one more entry than actions, rewards
		state = env.reset()
		states.append(state)
		for j in range(T):
			print (state)
			state = np.reshape(state, (1, -1))
			action = np.asscalar(self.policy_network.get_action(state))

			# the action chosen can be out of range, probably when NaNs occur from training
			# if action != 0 and action != 1:
			# 	print ('Wrong action: {}'. format(action))

			next_state, reward, done, _ = env.step(action)

			if render:
				env.render()

			states.append(next_state) # state is a standard list, not numpy array which would cause problems feeding dict
			actions.append(action)
			if done:
				if len(rewards) < 199:
					reward = -10 # CHANGE THIS ACOORDINGLY FOR CARTPOLE/MOUNTAINCAR
			rewards.append(reward)

			if done:
				break
			

			state = next_state
		env.close()

		return states, actions, rewards

		# states, actions, rewards = [], [], []
		# state = env.reset()
		# state = prepro(state)
		# prev_state = np.zeros_like(state)
		# x = state - prev_state
		# states.append(x)
		# for j in range(T):
		# 	x = state - prev_state
		# 	x = np.reshape(x, (1, -1))
		# 	action = np.asscalar(self.policy_network.get_action(x))
		# 	next_state, reward, done, _ = env.step(action)

		# 	env.render()

		# 	next_state = prepro(next_state)
		# 	new_x = next_state - state

		# 	states.append(new_x)
		# 	actions.append(action)
		# 	rewards.append(reward)

		# 	if done:
		# 		break

		# 	prev_state = state
		# 	state = next_state
		# env.close()

		# return states, actions, rewards

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

			states = np.reshape(states, (-1, len(states[0])))
			print (states.shape)
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
		target = np.asscalar(target)
		advantage = target - original_value

		return target, advantage


	def train(self, env, iterations, N, T, epochs, M, render):
		for i in range(iterations):
			print ('Collecting data...')
			state_data, action_data, reward_data = self.collect_data(env, N, T, render)

			average_reward = np.sum(np.sum(reward_data))/len(reward_data)
			print ('Iteration {}: Average reward = {}'.format(i, average_reward))

			print ('Computing targets and advantages...')
			target_data, advantage_data = self.compute_targets_and_advantages(state_data, reward_data)

			# self.print_for_debug()

			for j in range(epochs):
				batch_generator = self.get_batches(M, state_data, action_data, target_data, advantage_data)
				for s, a, t, adv in batch_generator:
					s = np.vstack(s) # COMMENT OUT THIS LINE IF NECESSARY (E.G. WHEN NOT RUNNING PONG)
					self.value_network.update(s, t)
					self.policy_network.update(s, a, adv)

			if i%10 == 0:
				self.policy_network.save_model()
				self.value_network.save_model()


	def get_batches(self, M, state_data, action_data, target_data, advantage_data): # concatenate the states, actions... lists together into one whole list
		# need to remove last entry for each state
		# flat_state_data = np.concatenate(tuple(states[:-1] for states in state_data), axis=0)
		# flat_action_data = np.concatenate(tuple(actions for actions in action_data), axis=0)
		# flat_target_data = np.concatenate(tuple(targets for targets in target_data), axis=0)
		# flat_advantage_data = np.concatenate(tuple(advantages for advantages in advantage_data), axis=0)

		flat_state_data = np.array([s for states in state_data for s in states[:-1]])
		flat_action_data = np.array([a for actions in action_data for a in actions])
		flat_target_data = np.array([t for targets in target_data for t in targets])
		flat_advantage_data = np.array([adv for advantages in advantage_data for adv in advantages])

		n_data = len(flat_state_data)
		randperm = random.sample(range(n_data), n_data)
		# randperm = np.arange(n_data)

		for i in range(0, n_data, M):
			end_i = min(i+M, n_data)
			ii = randperm[i:end_i]
			yield flat_state_data[ii], flat_action_data[ii], flat_target_data[ii], flat_advantage_data[ii]

	def save_model(self):
		save_name = self.saver.save(self.sess, self.save_path)
		print("Model checkpoint saved in file: %s" % save_name)

	def print_for_debug(self):
		self.policy_network.print_for_debug()
		self.value_network.print_for_debug()