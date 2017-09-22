import gym
import numpy as np

from PPOLearnerDiscrete import PPOLearnerDiscrete

env_name = 'Breakout-v0'
# env_name = 'PongDeterministic-v4'
# env_name = 'CartPole-v0'
env = gym.make(env_name)
# env = gym.wrappers.Monitor(env, env_name, force=True)

action_dim = env.action_space.n
obs = env.reset()
# state_dim = len(obs)
state_dim = 6400

print (action_dim)
print (state_dim)

iterations = 10000
epochs = 10
N = 10
T = 11000
M = 32
lam = 0.9
discount = 0.99
value_lr = 0.00005
policy_lr = 0.00005

render = True

learner = PPOLearnerDiscrete(action_dim, state_dim, discount, lam, value_lr, policy_lr)

learner.train(env, iterations, N, T, epochs, M, render)
