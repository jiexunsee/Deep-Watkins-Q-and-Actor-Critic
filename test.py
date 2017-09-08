import gym
import tensorflow as tf
import numpy as np

class Agent():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.sess = tf.InteractiveSession()
        self.s = tf.placeholder(tf.float32, [None, 4], name='s')
        self.a = tf.placeholder(tf.int32, [None], name='a')
        self.r = tf.placeholder(tf.float32, [None], name='r')
         
        self.W = tf.Variable(tf.random_normal([4, 2]), name='W')
        self.output = tf.nn.softmax(tf.matmul(self.s, self.W)) # simple linear policy
        one_hot = tf.one_hot(self.a, 2)
        o = tf.reduce_sum(self.output * one_hot, axis=1)
        self.loss = -tf.reduce_sum(tf.log(o) * self.r)
        
        Wg = tf.gradients(self.loss, [self.W])[0]
        solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = solver.apply_gradients([(Wg, self.W)])
        
    def train(self, s, a, r):
        self.sess.run(self.update, {self.s: s, self.a: a, self.r: r})
        
    def decide(self, s):
        o = self.sess.run(self.output, {self.s: [s]})[0]
        return 0 if np.random.random() <= o[0] else 1 # probabilistic policy

gamma = 0.95
n_episodes = 300
train_every = 5
tf.reset_default_graph()
agent = Agent(learning_rate=0.2)
tf.global_variables_initializer().run()

s_all, a_all, r_all = [], [], []
nits = []
from gym import wrappers
env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, '/home/ubuntu/gym_monitor', force=True)
for ei in range(n_episodes):
    s_arr, a_arr, r_arr = [], [], []
    s = env.reset()
    nit = 0
    done = False
    while not done:
        env.render()
        a = agent.decide(s)
        s1, r, done, _ = env.step(a)
        nit += 1
        s_arr.append(s)
        a_arr.append(int(a))
        r_arr.append(r)
        s = s1
    nits.append(nit)
    if ei % 10 == 0:
        print("Episode {} ran for {} steps".format(ei, np.mean(nits[-100:])))
    
    for i in reversed(range(nit)):
        if i + 1 < nit:
            r_arr[i] += gamma * r_arr[i+1]
    
    r_arr = list((r_arr - np.mean(r_arr)) / np.std(r_arr))
    if nit >= 200:
        r_arr = [1.0] * len(r_arr)
    nit = 0
    
    s_all += s_arr
    a_all += a_arr
    r_all += r_arr
        
    if ei > 0 and ei % train_every == 0:   
        agent.train(s_all, a_all, r_all)
        s_all, a_all, r_all = [], [], []
        
env.close()