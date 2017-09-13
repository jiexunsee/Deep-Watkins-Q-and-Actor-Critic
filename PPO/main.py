import gym
import numpy as np

from PPOLearner import PPOLearner
from PPOLearnerDiscrete import PPOLearnerDiscrete

''' Hopper stuff'''
# env = gym.make('Hopper-v1')

# action_dim = 
# state_dim = 

# iterations = 100
# epochs = 15
# T = 512
# M = 4096 # minibatch size
# discount = 0.995
# lam = 0.98
# N = 32 # actors

# value_lr = 1e-2 / np.sqrt(h2_size)
# policy_lr = 9e-4 / np.sqrt(h2_size)

''' Testing on simple problem: CartPole '''
env_name = 'Pong-v0'
# env_name = 'CartPole-v0'
env = gym.make(env_name)
# env = gym.wrappers.Monitor(env, env_name, force=True)

action_dim = env.action_space.n
obs = env.reset()
# state_dim = len(obs)
state_dim = 6400

print (action_dim)
print (state_dim)

iterations = 1000
epochs = 10
N = 1
T = 11000
M = 32
lam = 0.9
discount = 0.99
value_lr = 0.0005
policy_lr = 0.0005

render = True

learner = PPOLearnerDiscrete(action_dim, state_dim, discount, lam, value_lr, policy_lr)

learner.train(env, iterations, N, T, epochs, M, render)

# learner.save_model()