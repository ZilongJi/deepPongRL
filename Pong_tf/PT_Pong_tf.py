import gym
import copy
import numpy as np
import cPickle as pickle

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline  

import sys

# log the information to a text file

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.log.flush()

sys.stdout = Logger("tf_log.txt")

# define same basic CNN helfer functions

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
                        
# hyperparameters
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

# model initialization
image_size = 80 
D = image_size * image_size # input dimensionality: 80x80 grid

# model definition in tensorflow
tf.reset_default_graph()

observations = tf.placeholder(tf.float32, [None, D] , name="input_x")


x_image = tf.reshape(observations, [-1,image_size,image_size,1])

# define the first layer: convolution + ReLU
W_conv1 = tf.get_variable("W1", shape=[12,12,1,32], initializer=tf.contrib.layers.xavier_initializer())
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))
h_pool1 = max_pool_2x2(h_conv1)

# define the second layer: convolution + ReLU
W_conv2 = tf.get_variable("W2", shape=[8,8,32,48], initializer=tf.contrib.layers.xavier_initializer())
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2))
h_pool2 = max_pool_2x2(h_conv2)

# define the third layer: densely connected layer
W_fc1 = tf.get_variable("W3", shape=[20 * 20 * 48, 256], initializer=tf.contrib.layers.xavier_initializer())
h_pool2_flat = tf.reshape(h_pool2, [-1, 20*20*48])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1))

# softmax laye
W_fc2 = tf.get_variable("W4", shape=[256, 1], initializer=tf.contrib.layers.xavier_initializer())
y_conv = tf.matmul(h_fc1, W_fc2)

# now we get the probability of moving up
probability = tf.nn.sigmoid(y_conv)

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# The loss function. This sends the weights in the direction of making actions 
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loss = tf.reduce_mean((tf.square(input_y - probability) * advantages) )
newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=0.001) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_gradW1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_gradW2")
W3Grad = tf.placeholder(tf.float32,name="batch_gradW3")
W4Grad = tf.placeholder(tf.float32,name="batch_gradW4")
batchGrad = [W1Grad,W2Grad,W3Grad, W4Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,ys,drs = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

init = tf.initialize_all_variables()

sess =  tf.Session()
sess.run(init)
gradBuffer = sess.run(tvars)

for ix in range(len(gradBuffer)): gradBuffer[ix] = 0.0 * gradBuffer[ix]


saver = tf.train.Saver()

if resume:
    saver.restore(sess, "TF_RL_save.p")

won = 0
lost = 0

                       
env.monitor.start('/root/tmp/pong-experiment-1',force=1)
env.reset()

while episode_number <= 4000:

  if render: env.render()

  # Reset the gradient placeholder. We will collect gradients in 
  # gradBuffer until we are ready to update our policy network. 
  #gradBuffer = sess.run(tvars)


  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  x = x.reshape([1,80*80])

  # forward the policy network and sample an action from the returned probability
  tfprob = sess.run(probability,feed_dict={observations: x})

  action = 2 if np.random.uniform() < tfprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  #hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  ys.append(y)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if len(drs) > 3000:
    drs.pop(0)
    ys.pop(0)
    xs.pop(0)
       
  if reward != 0:
    if reward > 0: won+=1 
    else: lost+=1

  if done: # an episode finished
    
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    epy = np.vstack(ys)
    epr = np.vstack(drs)
    xs,ys,drs = [],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    
    # Get the gradient for this episode, and save it in the gradBuffer
    tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
    for ix,grad in enumerate(tGrad): gradBuffer[ix] += grad  

    # perform parameter update every batch_size episodes
    if episode_number % batch_size == 0:
           
        sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1], W3Grad: gradBuffer[2], W4Grad: gradBuffer[3]})#,W3Grad: gradBuffer[2],W4Grad:gradBuffer[3],W5Grad:gradBuffer[4]})

        # reset batch gradient buffer
        for ix in range(len(gradBuffer)): gradBuffer[ix] = 0.0 * gradBuffer[ix]

    # book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    if episode_number % 10 == 0: print 'resetting env. episode %d reward total was %f. running mean: %f, won: %.1f %%' % (episode_number, reward_sum, running_reward, 100.0*float(won)/float(won+lost))
    if episode_number % 100 == 0: save_path = saver.save(sess, "TF_RL_save.p")
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

env.monitor.close()

                        
sess.close()
plt.plot(step_count)                       
                        
                        
                        
                        
                        
