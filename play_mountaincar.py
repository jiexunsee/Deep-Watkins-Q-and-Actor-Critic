import gym
import numpy as np
import time

from helper import *
from PolicyLearner import PolicyLearner

name_of_gym = 'MountainCar-v0'
episodes = 50000

env = gym.make(name_of_gym)
env = gym.wrappers.Monitor(env, 'tmp/MountainCar-1', force=True)

n_actions = env.action_space.n
obs = env.reset()
n_states = len(obs)

agent = PolicyLearner(n_actions=n_actions, n_states=n_states, discount=0.95, lambda_w=0.5, lambda_theta=0.5, alpha=0.1, beta=0.01)

# Iterate the game
s = time.time()

for e in range(episodes):
	state = env.reset()

	total_reward = 0
	done = False
	while not done:
		state = np.reshape(state, (1, -1))
		action = agent.get_action(state)

		next_state, reward, done, _ = env.step(action)
		next_state = np.reshape(next_state, (1, -1))
		# env.render()

		if done:
			if total_reward > -199:
				reward = 10

		agent.learn(state, action, next_state, reward)

		state = next_state
		total_reward += reward
		
		if done:
			print("episode: {}/{}, score: {:.0f}".format(e, episodes, total_reward))

	# if e%100 == 0:
		# agent.print_for_debug()
	# 	print("episode: {}/{}, score: {:.0f}".format(e, episodes, total_reward))
	
	agent.reset_e_trace()

e = time.time()
print ('TIME TAKEN: {}'.format(e-s))
env.close()

# gym.upload('tmp/MountainCar-1', api_key='sk_5iXTcYwRUy9chDqhy4M6w')