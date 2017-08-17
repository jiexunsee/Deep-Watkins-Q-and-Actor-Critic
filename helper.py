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
		else:
			print ('FOUND GOAL!')
	return reward

def package_state(state, name_of_gym):
	if name_of_gym == 'FrozenLake-v0':
		state = convert_to_one_hot(state, 16)
		state = state.reshape(1, -1)
		return state
	elif name_of_gym == 'CartPole-v0':
		return state

def convert_to_one_hot(state_number, n_states):
	state = np.zeros((1,n_states))
	state[0][state_number] = 1
	return state