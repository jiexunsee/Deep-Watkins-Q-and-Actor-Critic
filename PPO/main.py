import gym
import numpy as np

from PPOLearner import PPOLearner

env = gym.make('Hopper-v1')

action_dim = 
state_dim = 

iterations = 100
epochs = 15
T = 512
M = 4096 # minibatch size
discount = 0.995
lam = 0.98
N = 32 # actors

value_lr = 1e-2 / np.sqrt(h2_size)
policy_lr = 9e-4 / np.sqrt(h2_size)

learner = PPOLearner(action_dim, state_dim, discount, lam, value_lr, policy_lr)

PPOLearner.train(env, iterations, N, T, epochs, M)