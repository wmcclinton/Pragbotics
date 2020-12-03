"""TODO(wmcclinton): DO NOT SUBMIT without one-line documentation for test.

TODO(wmcclinton): DO NOT SUBMIT without a detailed description of test.
"""

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np

import gym
from monitoring import VideoRecorder
import getch

import random
import pickle
from pong import *

import time

FLAGS = flags.FLAGS

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

env = get_pong_env()

# Init Random Policy (NN) given (Env)
def gen_rand_policy():
  num_actions = env.action_space.n

  policy = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='sigmoid')
  ])

  return policy

def get_policy_action(policy, obs):
  action = np.argmax(policy(np.array([obs.flatten()])).numpy()[0])
  return action

def get_user_action(obs):
  num_actions = env.action_space.n

  print('Obs',obs)
  action = -1
  while not (action >= 0 and action < num_actions):
    try:
      print(env.unwrapped.get_action_meanings())
      print('\naction[0-' + str(num_actions-1) + '] << ')
      action = int(getch.getche())
      print()
    except:
      action = -1

  return action

# Sample Trajectory (array and vid) given (reset Env)
def gen_sample_traj(policy, max_steps=100, path=None, record=False):
  # if record Save Policy and Trajectories to Folder
  video_path = path
  video_recorder = None
  if record:
    video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)

  traj = []
  obs = env.reset()
  obs = get_pong_symbolic(obs)

  for i in range(max_steps):
    _obs = obs
    action = get_policy_action(policy, obs)
    obs, reward, terminate, _ = env.step(action)
    obs = get_pong_symbolic(obs)
    traj.append([_obs, action, reward, terminate])

    #print(traj[-1])
    #env.render()  # Note: rendering increases step time.

    if record:
      video_recorder.capture_frame()

    if terminate:
      print('Total Steps:', i)
      break

  if record:
    video_recorder.close()

  return traj

# Get Demonstrations from user
def gen_user_traj(max_steps=100, path=None, record=False):
  # if record Save Policy and Trajectories to Folder
  video_path = path
  video_recorder = None
  if record:
    video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)

  traj = []
  obs = env.reset()
  obs = get_pong_symbolic(obs)

  for i in range(max_steps):
    _obs = obs
    action = get_user_action(obs)
    obs, reward, terminate, _ = env.step(action)
    obs = get_pong_symbolic(obs)
    traj.append([_obs, action, reward, terminate])

    #print(traj[-1])
    env.render()  # Note: rendering increases step time.

    if record:
      video_recorder.capture_frame()

    if terminate:
      print('Total Steps:', i)
      break

  if record:
    video_recorder.close()

  return traj

def one_hot(idx, num_classes):
  a = np.zeros(num_classes)
  a[idx] = 1
  return a

# Create Function to fit to Demonstrations
def train_policy(demos, policy):
  # TODO Warning!!! Probably does not work
  # Format demos into x and y data
  x = []
  y = []
  num_actions = env.action_space.n
  for demo in demos:
    x = x + [t[0] for t in demo[0]]
    y = y + [one_hot(t[1], num_actions) for t in demo[0]]

  print(np.array(x).shape)
  print(np.array(y).shape)

  z = list(zip(x, y))

  random.shuffle(z)

  x, y = zip(*z)

  x = np.array(x)
  y = np.array(y)

  # Fit to demos
  loss_fn = tf.keras.losses.MeanSquaredError()

  policy.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['MeanSquaredError'])

  policy.fit(x, y, epochs=1000, verbose=1)

  # Evaluate?
  policy.evaluate(x, y, verbose=2)

  return policy

# Save Policy
def save_policy(policy, path):
  print('Saved to',path)
  policy.save(path)
  return path

# Load Policy
def load_policy(path):
  print('Loaded from',path)
  policy = tf.keras.models.load_model(path)
  return policy

# Display Policy
def print_policy(policy):
  weights = policy.get_weights()
  print(weights[0])
  return weights


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  #(1) make win as soon as possible
  #(2) keep rally as long as possible
  #(3) lose as soon as possible

  num_demos = 10
  demos = []
  for n in range(num_demos):
    print('Getting Demo ' + str(n) + '...')
    demos.append([gen_user_traj(max_steps=500, path='demos/willie_demo_' + str(int(time.time())) + '.mp4', record=True)])

  pickle.dump(demos, open( "demos_" + str(int(time.time())) + ".p", "wb"))

if __name__ == '__main__':
  #print('\nHello Google!')
  app.run(main)
