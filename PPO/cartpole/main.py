import gym
import numpy as np

from PPOLearnerCartPole import PPOLearnerDiscrete

''' Testing on simple problem: CartPole '''
env_name = 'CartPole-v1'
env = gym.make(env_name)
env = gym.wrappers.Monitor(env, env_name, force=True)

action_dim = env.action_space.n
obs = env.reset()
state_dim = len(obs)

iterations = 1000
epochs = 10
N = 5
T = 1100
M = 32
lam = 0.9
discount = 0.9
value_lr = 0.04
policy_lr = 0.04

render = True

learner = PPOLearnerDiscrete(action_dim, state_dim, discount, lam, value_lr, policy_lr)

learner.train(env, iterations, N, T, epochs, M, render)