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

FLAGS = flags.FLAGS

seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

# Init Random Policy (NN) given (Env)
def gen_rand_policy(env):
  num_actions = env.action_space.n

  policy = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='sigmoid')
  ])

  return policy

def get_policy_action(policy, obs):
  action = np.argmax(policy(np.array([obs.flatten()])).numpy()[0])
  return action

def get_user_action(env, obs):
  num_actions = env.action_space.n

  print('Obs',obs)
  action = -1
  while not (action >= 0 and action < num_actions):
    try:
      print('\naction[0-' + str(num_actions) + '] << ')
      action = int(getch.getche())
      print()
    except:
      action = -1

  return action

# Sample Trajectory (array and vid) given (reset Env)
def gen_sample_traj(env, policy, max_steps=100, path=None, record=False):
  # if record Save Policy and Trajectories to Folder
  video_path = path
  video_recorder = None
  if record:
    video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)

  traj = []
  obs = env.reset()

  for i in range(max_steps):
    _obs = obs
    action = get_policy_action(policy, obs)
    obs, reward, terminate, _ = env.step(action)
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
def gen_user_traj(env, max_steps=100, path=None, record=False):
  # if record Save Policy and Trajectories to Folder
  video_path = path
  video_recorder = None
  if record:
    video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)

  traj = []
  obs = env.reset()

  for i in range(max_steps):
    _obs = obs
    action = get_user_action(env, obs)
    obs, reward, terminate, _ = env.step(action)
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

def one_hot(idx, num_classes):
  a = np.zeros(num_classes)
  a[idx] = 1
  return a

# Create Function to fit to Demonstrations
def train_policy(env, demos, policy):
  # TODO Warning!!! Probably does not work
  # Format demos into x and y data
  x = []
  y = []
  num_actions = env.action_space.n

  for demo in demos:
    x = x + [t[0].flatten() for t in demo]
    y = y + [one_hot(t[1], num_actions) for t in demo]

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

  policy.fit(x, y, epochs=10, verbose=1)

  # Evaluate?
  policy.evaluate(x,  y, verbose=2)

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

  env = gym.make('Pong-v0')

  load_agents = False

  if load_agents:
    policy = load_policy('agents/policy1')
    traj = gen_sample_traj(env, policy, max_steps=100, path='vids/video.mp4', record=True)
    #print(traj)
  else:
    policy = gen_rand_policy(env)
    traj = gen_sample_traj(env, policy, max_steps=100, path='vids/video.mp4', record=True)
    #print(traj)

    save_policy(policy, 'agents/policy1')

  print('Getting Demo 1...')
  demo1 = gen_user_traj(env, max_steps=100, path='vids/demo1.mp4', record=True)
  #print(traj)

  print('Getting Demo 2...')
  demo2 = gen_user_traj(env, max_steps=100, path='vids/demo2.mp4', record=True)
  #print(traj)

  # TODO fit to demos (WARNING!!!: prob doesn't work)
  demos = [demo1, demo2]
  train_policy(env, demos, policy)

  ###

  #save_policy(policy, 'agents/*bad*_trained_policy1')

  print('Done')
  env.close()


if __name__ == '__main__':
  print('\nHello Google!')
  app.run(main)
