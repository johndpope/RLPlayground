# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import time
import tensorflow as tf

import model

parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=int, default=5)
parser.add_argument("--sequence_length", type=int, default=5)
parser.add_argument("--num_train_steps", type=int, default=1000000)
parser.add_argument("--eval_intervals", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--temperature", type=float, default=1.)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--regularization", type=float, default=1e-5)
args = parser.parse_args()


class GameState():
  def __init__(self):
    self.source, self.target = np.random.permutation(args.world_size)[:2]
    self.world = np.zeros([args.world_size], dtype=np.int32)
    self.world[self.source] = 1
    self.world[self.target] = 2 
    self.time = int(abs(self.target - self.source) * 2)

  def Play(self, act):
    new_pos = self.source + [-1, 1][act] 
    self.world[self.source] = 0
    self.source = new_pos
    if self.IsValid(new_pos):
      self.world[new_pos] = 1
    self.time = max(0, self.time - 1)

  def IsWon(self):
    return self.source == self.target

  def IsLost(self):
    return not self.IsValid(self.source) or self.time <= 0

  def IsEnded(self):
    return self.IsWon() or self.IsLost()

  def GetReward(self):
    return 1 if self.source == self.target else -1

  def IsValid(self, pos):
    return pos >= 0 and pos < self.world.size


def Policy(m, observations, stochastic=True):
  return m.outputs.eval(feed_dict={
    m.observations: observations,
    m.temperature: args.temperature})


def PrintGameState(state):
  print " ".join([[u" ", u"☺", u"◈"][i] for i in state.world])


def PlaySampleGame(m):
  state = GameState()
  PrintGameState(state)
  while not state.IsEnded():
    p = Policy(m, state.world[np.newaxis])[0]
    act = np.argmax(p)
    state.Play(act)
    PrintGameState(state)
  print "Won." if state.IsWon() else "Lost." 


class Data(object):
  def __init__(self):
    self.observations = []
    self.actions = []
    self.rewards = []


def GenerateData(model):
  states = []
  data = []
  for _ in range(args.batch_size):
    states.append(GameState())
    data.append(Data())

  # Play with current policy
  observations = np.zeros([args.batch_size, args.world_size], dtype=np.int32)
  for _ in range(args.sequence_length):
    for i, state in enumerate(states):
      observations[i] = state.world
    ps = Policy(model, observations)
    all_ended = True
    for i, (d, state) in enumerate(zip(data, states)):
      p = ps[i]
      action = np.random.choice(ps[i].size, 1, p=p/p.sum())[0]
      d.observations.append(observations[i])
      d.actions.append(action)
      if not state.IsEnded():
        state.Play(action)
        all_ended = False
    if all_ended:
      break

  # Update the rewards
  for i, state in enumerate(states):
    r = state.GetReward()
    seq_length = len(data[i].actions)
    data[i].rewards = r * np.ones(seq_length, dtype=np.float32)

  # Concatenate all the data
  cat_data = Data()
  cat_data.observations = np.concatenate([d.observations for d in data])
  cat_data.actions = np.concatenate([d.actions for d in data])
  cat_data.rewards = np.concatenate([d.rewards for d in data])
  return cat_data


def Train():
  m = model.Model(input_dim=args.world_size, embedding_rows=3, 
    embedding_cols=3, output_dim=2,  hidden_dims=[10], 
    lr=args.learning_rate, reg_factor=args.regularization)
    
  sess = tf.Session()
  saver = tf.train.Saver()

  with sess.as_default():
    sess.run(tf.initialize_all_variables())

    last_time = time.time()
    global_step = m.global_step.eval()
    while global_step < args.num_train_steps:
      data = GenerateData(m)
      loss, global_step, _ = sess.run(
        [m.loss, m.global_step, m.optimize],
        feed_dict={
          m.observations: data.observations,
          m.actions: data.actions,
          m.rewards: data.rewards,
          m.temperature: args.temperature
        })

      if global_step % args.eval_intervals == 0:
        cur_time = time.time()
        print("Loss at global step %d is %f, took %.2f seconds." % 
          (global_step, loss, cur_time-last_time))
        last_time = cur_time
        PlaySampleGame(m)


if __name__ == "__main__":
  Train()
