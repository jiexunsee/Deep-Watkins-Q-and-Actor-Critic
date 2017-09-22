import gym
from gym import wrappers
import numpy as np
import time
import sys

from helper import *
from DeepTDLambdaLearner import DeepTDLambdaLearner
from PolicyLearner import PolicyLearner

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

if len(sys.argv) <= 1:
	print ('Using lambda of 0.5')
	lamb = 0.5
else:
	lamb = float(sys.argv[1])


name_of_gym = 'FrozenLake-v0'
episodes = 600

env = gym.make(name_of_gym)
# env = gym.make('FrozenLakeNotSlippery-v0')
# env = wrappers.Monitor(env, '/tmp/frozenlake-1', force=True)
n_actions = env.action_space.n

try:
	n_states = env.observation_space.n
except:
	obs = env.reset()
	n_states = len(obs)
	
agent = DeepTDLambdaLearner(n_actions=n_actions, n_states=n_states, discount=0.95, alpha=0.05, epsilon=1, epsilon_decay=0.99, lamb=lamb)

# Iterate the game

s = time.time()
success = 0

for e in range(episodes):
	state = env.reset()
	state = package_state(state, name_of_gym)

	total_reward = 0
	done = False
	while not done:
		action, greedy = agent.get_e_greedy_action(state)
		next_state, reward, done, _ = env.step(action)
		# env.render()
		
		next_state = package_state(next_state, name_of_gym)
	
		# Tweaking the reward to help the agent learn faster
		tweaked_reward = tweak_reward(reward, done, name_of_gym)
		
		agent.learn(state, action, next_state, tweaked_reward, greedy)

		state = next_state
		total_reward += tweaked_reward
		
		if done:
			if reward == 1:
				print("episode: {}/{}, score: {:.2f} and goal has been found!".format(e, episodes, total_reward))
				if e >= episodes - 100:
					success += 1
			else:
				print("episode: {}/{}, score: {:.2f}".format(e, episodes, total_reward))
			break
	
	# if e%100 == 0:
	# 	agent.print_weights()

	agent.reset_e_trace()

agent.print_weights()

e = time.time()
print ('TIME TAKEN: {}'.format(e-s))
print ('RECENT SUCCESS RATE: {}/{}'.format(success, 100))
# env.close()

# gym.upload('/tmp/frozenlake-1', api_key='sk_5iXTcYwRUy9chDqhy4M6w')