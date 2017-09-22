from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

import gym

env = gym.make('FrozenLakeNotSlippery-v0')

s = env.reset()
next_state, reward, done, _ = env.step(0)
print (next_state)

next_state, reward, done, _ = env.step(1)
print (next_state)
next_state, reward, done, _ = env.step(1)
print (next_state)
next_state, reward, done, _ = env.step(2)
print (next_state)
next_state, reward, done, _ = env.step(2)
print (next_state)
next_state, reward, done, _ = env.step(1)
print (next_state)
next_state, reward, done, _ = env.step(0)
print (next_state)
next_state, reward, done, _ = env.step(2)
print (next_state)
next_state, reward, done, _ = env.step(2)
print (next_state)
print ('reward')
print (reward)

# 1-down, 3-up, 2-right, 0-left
# left, down, right, up