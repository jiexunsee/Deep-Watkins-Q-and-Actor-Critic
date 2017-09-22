import gym
import numpy as np

from PPOLearner import PPOLearner
from PPOLearnerDiscrete import PPOLearnerDiscrete

''' Hopper stuff'''
# env = gym.make('Hopper-v1')
# obs = env.reset()

# action_dim = 3
# state_dim = len(obs)

# iterations = 500000
# epochs = 15
# T = 512
# M = 4096 # minibatch size
# discount = 0.99
# lam = 0.98
# N = 32 # actors


# v_h1_size = state_dim*10
# v_h3_size = 5
# v_h2_size = np.sqrt(v_h1_size*v_h3_size)

# p_h1_size = state_dim*10
# p_h3_size = action_dim*10
# p_h2_size = int(np.sqrt(p_h1_size*p_h3_size))

# value_lr = 1e-2 / np.sqrt(v_h2_size)
# policy_lr = 9e-4 / np.sqrt(p_h2_size)

# learner = PPOLearner(action_dim, state_dim, discount, lam, value_lr, policy_lr, 'model/value_network', 'model/policy_network')

# learner.train(env, iterations, N, T, epochs, M)


''' Testing on Pendulum '''
# env_name = 'Pendulum-v0'
# env = gym.make(env_name)
# # env = gym.wrappers.Monitor(env, env_name, force=True)

# action_dim = env.action_space.shape[0] # assuming action space is just Box(1,)
# obs = env.reset()
# state_dim = len(obs)

# print (action_dim)
# print (state_dim)

# iterations = 1000
# epochs = 1
# N = 1
# T = 11000
# M = 32
# lam = 0.9
# discount = 0.99
# value_lr = 0.001
# policy_lr = 0.0001

# render = True

# learner = PPOLearner(action_dim, state_dim, discount, lam, value_lr, policy_lr, 'pendulum_model/value_network', 'pendulum_model/policy_network')

# learner.train(env, iterations, N, T, epochs, M)

''' Testing on InvertedPendulum '''
''' Keeps failing immediately, after 64/65 actions. '''
env_name = 'InvertedPendulum-v1'
env = gym.make(env_name)
# env = gym.wrappers.Monitor(env, env_name, force=True)

action_dim = env.action_space.shape[0] # assuming action space is just Box(1,)
obs = env.reset()
state_dim = len(obs)

print (action_dim)
print (state_dim)

iterations = 100000
epochs = 15
N = 32
T = 11000
M = 1024
lam = 0.9
discount = 0.99
value_lr = 0.00001
policy_lr = 0.00001

render = True

learner = PPOLearner(action_dim, state_dim, discount, lam, value_lr, policy_lr, 'invertedpendulum_model/value_network', 'invertedpendulum_model/policy_network')

learner.train(env, iterations, N, T, epochs, M)