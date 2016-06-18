from __future__ import division, print_function

import argparse
import numpy as np
import os
import time
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--num_train_steps", type=int, default=1000000)
parser.add_argument("--num_turns", type=int, default=5)
parser.add_argument("--num_arms", type=int, default=10)
parser.add_argument("--max_game_steps", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--regularization", type=float, default=0.0001)
args = parser.parse_args()


class Model(object):
  def __init__(self, num_actions, lr, reg_factor):
    # Define inputs 
    self.actions = tf.placeholder(tf.int32, shape=(None))
    self.rewards = tf.placeholder(tf.float32, shape=(None))
    self.batch_size = tf.placeholder_with_default(1, [])

    # Calculate probabilities for all actions
    self.weights = tf.get_variable("weights", [num_actions])
    logits = tf.tile(
      tf.reshape(self.weights, [1, -1]),
      tf.pack([self.batch_size, 1]))
    self.outputs = tf.nn.softmax(logits) 

    # Calculate log-likelihoods for given action 
    lls = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, self.actions)
    
    # Calculate weight regularization term
    reg_term = tf.add_n(
      [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    
    # Maximize stochastic expectation of rewards   
    self.loss = tf.reduce_mean(self.rewards * lls) + reg_term * reg_factor
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    # Calculate gradients and subtract from the weights
    self.gradients = tf.gradients(self.loss, self.weights)[0]
    self.optimize = self.weights.assign_sub(self.gradients * lr)


class GameState():
  rewards = np.arange(args.num_arms).astype(np.float32)
  
  def __init__(self):
    self.time = args.num_turns

  def Play(self, act): 
    self.time = max(0, self.time - 1)
    return self.rewards[act]

  def GetBestAction(self):
    return np.argmax(self.rewards)

  def IsEnded(self):
    return self.time == 0 


def PlayTurn(m, state, stochastic=True):
  p = m.outputs.eval()[0]
  if stochastic:
    act = np.random.choice(p.size, 1, p=p/p.sum())[0]
  else:
    act = np.argmax(p)
  reward = state.Play(act)
  return act, reward 


def PrintWeights(m):
  weights = m.weights.eval()
  print("Weights:", weights)
  print("Softmax:", np.exp(-weights)/np.exp(-weights).sum())


def PlaySampleGame(m):
  PrintWeights(m)
  state = GameState()
  print("Best action:", state.GetBestAction())
  rewards = []
  actions = []
  while not state.IsEnded():
    action, reward = PlayTurn(m, state, stochastic=False)
    actions.append(action)
    rewards.append(reward)
  print("Taken actions:", np.array(actions))
  print("Collected rewards:", rewards, ">", np.array(rewards).sum())


class Data(object):
  def __init__(self):
    self.actions = []
    self.rewards = []


def GenerateData(model):
  # Generate complete games
  all_data = []
  for _ in range(args.batch_size):
    state = GameState()
    data = Data()

    # Play with current policy
    while not state.IsEnded():
      action, reward = PlayTurn(model, state)
      data.actions.append(action)
      data.rewards.append(reward)

    # Convert data to numpy and update the rewards
    data.actions = np.array(data.actions, dtype=np.int32)
    data.rewards = np.array(data.rewards, dtype=np.float32)
    all_data.append(data)
  
  # Concatenate all the data
  cat_data = Data()
  cat_data.actions = np.concatenate([d.actions for d in all_data])
  cat_data.rewards = np.concatenate([d.rewards for d in all_data])
  return cat_data


def Train():
  m = Model(num_actions=args.num_arms, lr=args.learning_rate, 
    reg_factor=args.regularization)
    
  sess = tf.Session()
  saver = tf.train.Saver()

  with sess.as_default():
    sess.run(tf.initialize_all_variables())

    last_time = time.time()

    print("Initial state:")
    PrintWeights(m)
    print()

    for step in range(args.num_train_steps):
      data = GenerateData(m)
      loss, weights, grads, _ = sess.run(
        [m.loss, m.weights, m.gradients, m.optimize],
        feed_dict={
          m.batch_size: data.actions.shape[0],
          m.actions: data.actions,
          m.rewards: data.rewards
        })

      cur_time = time.time()
      print("Loss at training step %d is %f, took %.2f seconds." % 
        (step, loss, cur_time-last_time))
      print("Gradients:", grads)
      PlaySampleGame(m)
      print()
      last_time = cur_time
      step += 1

      if all(np.equal(
        np.argsort(m.weights), 
        np.argsort(GameState.rewards))):
        print("Found optimal weights at step:", step)
        break


if __name__ == "__main__":
  Train()
