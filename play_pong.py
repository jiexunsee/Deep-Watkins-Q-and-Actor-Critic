import gym
import numpy as np
import time

from helper import *
from PolicyLearnerHidden import PolicyLearnerHidden

name_of_gym = 'Pong-v0'
episodes = 15000

env = gym.make(name_of_gym)
env = gym.wrappers.Monitor(env, 'tmp/Pong-1', force=True)

n_actions = env.action_space.n
obs = env.reset()
obs = prepro(obs)
n_states = len(obs)

agent = PolicyLearnerHidden(n_actions=n_actions, n_states=n_states, discount=0.8, lambda_w=0.3, lambda_theta=0.3, save_path='tmp/model.ckpt')

s = time.time()

for e in range(episodes):
	state = env.reset()
	state = prepro(state)
	prev_state = np.zeros_like(state)
	total_reward = 0
	done = False
	while not done:
		x = state - prev_state
		x = x.reshape(1, -1)
		
		action = agent.get_action(x)
		
		next_state, reward, done, _ = env.step(action)
		next_state = prepro(next_state)
		new_x = next_state - state
		new_x = new_x.reshape(1, -1)

		# env.render()

		agent.learn(x, action, new_x, reward)

		prev_state = state
		state = next_state
		total_reward += reward
		
		if done:
			print("episode: {}/{}, score: {:.0f}".format(e, episodes, total_reward))

	agent.reset_e_trace()

	if e%100 == 0:
		agent.save_model()

agent.save_model()

e = time.time()
print ('TIME TAKEN: {}'.format(e-s))
env.close()

# gym.upload('tmp/Pong-1', api_key='sk_5iXTcYwRUy9chDqhy4M6w')