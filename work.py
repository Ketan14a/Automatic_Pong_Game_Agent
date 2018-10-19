import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from keras.layers import Dense, Input, Reshape
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam, Adamax, RMSprop
import cv2
import gym

# Script Parameters
INPUT_DIM = 80 * 80
GAMMA = 0.99
DECAY_RATE = 0.99
NEURAL_NODES = 200      # for flat
UPDATE_FREQUENCY = 10
LEARNING_RATE = 1e-3
REWARD_SUM = 0
RESUME = True
RENDER = True

# Initialize
env = gym.make("Pong-v0")
n_actions = 2
observation = env.reset()
prev_x = None
# observation, gradient, reward record, and prob. record
xs, dlogps, drs, probs = [], [], [], []
running_reward = None
episode_number = 0
train_X = []
train_y = []


def pong_preprocess_screen(frame):
  frame = frame[35:195]                         # crop frame
  frame = frame[::2, ::2, 0]                    # halves frame & remove color
  frame[(frame == 144) | (frame == 109)] = 0    # remove background
  frame[frame != 0] = 1                         # set paddles and ball to 1
  return frame.astype(np.float).ravel()


def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0:
      running_add = 0
    running_add = running_add * GAMMA + r[t]
    discounted_r[t] = running_add
  return discounted_r


def learning_model(INPUT_DIM=80 * 80, model_type='flat'):
  model = Sequential()

  if model_type == 'flat':
    model.add(Reshape((1, 80, 80), input_shape=(INPUT_DIM,)))
    model.add(Flatten())
    model.add(Dense(NEURAL_NODES, activation='relu'))
    model.add(Dense(n_actions, activation='softmax'))
    opt = RMSprop(lr=LEARNING_RATE)
  elif model_type == 'CNN':
    model.add(Reshape((1, 80, 80), input_shape=(INPUT_DIM,)))
    model.add(Convolution2D(32, 9, 9, subsample=(4, 4),
                            border_mode='same', activation='relu', init='he_uniform'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu', init='he_uniform'))
    model.add(Dense(n_actions, activation='softmax'))
    opt = Adam(lr=LEARNING_RATE)

  model.compile(loss='categorical_crossentropy', optimizer=opt)

  if RESUME == True:
    model.load_weights('pong_model_checkpoint_%s.h5' % model_type)

  return model

if __name__ == '__main__':
  model_type = 'flat'
  model = learning_model(model_type=model_type)

  while 'Ping' != 'Pong':
    if RENDER:
      env.render()

    cur_x = pong_preprocess_screen(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(INPUT_DIM)
    prev_x = cur_x

    # Predict the probability of going up
    prob = ((model.predict(x.reshape([1, x.shape[0]]), batch_size=1)).flatten())

    xs.append(x)
    probs.append(prob)

    action = np.random.choice(n_actions, 1, p=prob)[0]
    y = np.zeros([n_actions])
    y[action] = 1

    # grad that encourages the action that was taken to be taken
    # http://cs231n.github.io/neural-networks-2/#losses
    dlogps.append(np.array(y).astype('float32') - prob)

    observation, reward, done, info = env.step(2 if action == 0 else 3)
    REWARD_SUM += reward
    drs.append(reward)

    if done:
      episode_number += 1

      # stack together all inputs, action gradients, and rewards for this eps.
      epx = np.vstack(xs)
      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)

      # compute reward through time and standardize it
      discounted_epr = discount_rewards(epr)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)

      # poilicy gradient: modulate the gradient with advantage
      epdlogp *= discounted_epr

      # append to the batch for later training
      train_X.append(xs)
      train_y.append(epdlogp)
      xs, dlogps, drs = [], [], []

      if episode_number % UPDATE_FREQUENCY == 0:
        y_train = probs + LEARNING_RATE * np.squeeze(np.vstack(train_y))
        model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)

        # Clear the batch
        train_X = []
        train_y = []
        probs = []

        model.save_weights('pong_model_checkpoint_%s.h5' % model_type, overwrite=True)

      running_reward = REWARD_SUM \
          if running_reward is None \
          else running_reward * DECAY_RATE + REWARD_SUM * (1 - DECAY_RATE)

      print("Episode %i. Total Reward: %f. Running Mean: %f" % (episode_number, REWARD_SUM, running_reward))
      REWARD_SUM = 0
      observation = env.reset()
      prev_x = None