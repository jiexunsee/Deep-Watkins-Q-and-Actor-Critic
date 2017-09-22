import gym
from gym import wrappers
import numpy as np
import time

from helper import *
# from DeepTDLambdaLearner import DeepTDLambdaLearner
from PolicyLearner import PolicyLearner
from PolicyLearnerHidden import PolicyLearnerHidden

# tf random seed 1000 works well

name_of_gym = 'CartPole-v1'
episodes = 10000

env = gym.make(name_of_gym)
env = wrappers.Monitor(env, 'tmp/cartpole-3', force=True)
n_actions = env.action_space.n

obs = env.reset()
n_states = len(obs)

# agent = PolicyLearnerHidden(n_actions=n_actions, n_states=n_states, lambda_w=0.5, lambda_theta=0.5, lstm_size=5, hidden=5, alpha=0.05, beta=0.001)
agent = PolicyLearner(n_actions=n_actions, n_states=n_states, lambda_w=0.9, lambda_theta=0.9, alpha=0.03, beta=0.005, discount=0.95)

# Iterate the game
s = time.time()

for e in range(episodes):
	state = env.reset()

	total_reward = 0
	done = False

	while not done:
		# agent.print_for_debug()
		state = np.reshape(state, (1, -1))
		action = agent.get_action(state)
		# action = np.asscalar(action)
		# print (action)
		next_state, reward, done, _ = env.step(action)
		next_state = np.reshape(next_state, (1, -1))
		env.render()

		if done:
			if total_reward < 499:
				reward = -10

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

# gym.upload('tmp/cartpole-3', api_key='sk_5iXTcYwRUy9chDqhy4M6w')