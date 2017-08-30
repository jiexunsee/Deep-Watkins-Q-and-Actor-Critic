import numpy as np


def tweak_reward(reward, done, name_of_gym):
	if name_of_gym == 'FrozenLake-v0':
		reward = tweak_frozen_lake_reward(reward, done)
		return reward
	elif name_of_gym == 'CartPole-v0':
		return reward
	else:
		print ('No reward tweaking implemented yet')
		return reward

def tweak_frozen_lake_reward(reward, done):
	if reward == 0:
		reward = -0.01
	if done:
		if reward < 1:
			reward = -1

	return reward

def package_state(state, name_of_gym):
	if name_of_gym == 'FrozenLake-v0':
		state = convert_to_one_hot(state, 16)
		state = state.reshape(1, -1)
		return state
	elif name_of_gym == 'CartPole-v0':
		return state.reshape(1, -1)

def convert_to_one_hot(state_number, n_states):
	state = np.zeros((1,n_states))
	state[0][state_number] = 1
	return state

class StateQueue:
	def __init__(self, initial_state, queue_length):
		self.queue_length = queue_length
		self.n_states = len(initial_state)
		initial_state = np.reshape(initial_state, (1, -1))
		zero_padding = np.zeros((1, self.n_states*(queue_length-1)))
		self.queue = np.concatenate((zero_padding, initial_state), axis=1)
		assert(self.queue.shape == (1, self.n_states*self.queue_length))

	def add_to_queue(self, new_state):
		new_state = np.reshape(new_state, (1, -1))
		old_queue = self.queue

		# print (self.queue)
		# print (self.queue[:,self.n_states:].shape)
		# print (new_state.shape)

		self.queue = np.concatenate((self.queue[:,self.n_states:], new_state), axis=1)
		assert(self.queue.shape == (1, self.n_states*self.queue_length))
		return old_queue, self.queue

def prepro(I):
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	I = I[35:195] # crop
	I = I[::2,::2,0] # downsample by factor of 2
	I[I == 144] = 0 # erase background (background type 1)
	I[I == 109] = 0 # erase background (background type 2)
	I[I != 0] = 1 # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel()