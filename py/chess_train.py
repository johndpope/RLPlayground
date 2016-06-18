import argparse
import numpy as np
import os
import time
import tensorflow as tf

import model
import chess
import chess_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model/")
parser.add_argument("--num_train_steps", type=int, default=1000000)
parser.add_argument("--checkpoint_intervals", type=int, default=1)
parser.add_argument("--max_game_steps", type=int, default=250)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--gamma", type=float, default=.99)
parser.add_argument("--temperature", type=float, nargs=2, default=[1., 1.])
args = parser.parse_args()


COLORS = ["black", "white"]


class Data(object):
  def __init__(self):
    self.observations = []
    self.actions = []
    self.rewards = []


def GenerateData(models):
  games = []
  data = []
  steps = []
  for _ in range(args.batch_size * 10):
    games.append(chess.Game())
    data.append([Data(), Data()])
    steps.append([0])

  # Play with current policy
  c = 1
  observations = np.zeros([len(games), 64], dtype=np.int32)
  while True:
    for step, d, game in zip(steps, data, games):
      if game.IsEnded() or step[0] == args.max_game_steps:
        if c == game.GetState().turn and game.IsCheckmate():
          print "%s won at %d steps." % (COLORS[c], step[0])
          yield d, game
        if c == 1:
          d[0] = Data()
          d[1] = Data()
          game.Reset()
          step[0] = 0
      step[0] += 1
    observations, actions, rewards = chess_utils.PlayTurn(
      models[c], games, args.temperature[c])
    for i, d in enumerate(data):
      d[c].observations.append(observations[i])
      d[c].actions.append(actions[i])
      d[c].rewards.append(rewards[i])
    c = 1 - c

def ProcessData(iterator):
  data = [Data(), Data()]
  for d, game in iterator:
    # Convert data to numpy and update the rewards
    for c in range(2):
      n = len(d[c].actions)
      r = -1 if game.GetState().turn == c else 1.
      data[c].observations.append(
        np.array(d[c].observations, dtype=np.int32))
      data[c].actions.append(
        np.array(d[c].actions, dtype=np.int32))
      data[c].rewards.append(
        r * np.logspace(n-1, 0, num=n, base=args.gamma, 
        dtype=np.float32))
    
    # Concatenate all the data
    if len(data[0].observations) == args.batch_size:
      cat_data = Data(), Data()      
      for c in range(2):    
        cat_data[c].observations = np.concatenate(data[c].observations)
        cat_data[c].actions = np.concatenate(data[c].actions)
        cat_data[c].rewards = np.concatenate(data[c].rewards)
        data[c] = Data()
      yield cat_data 


def Train():
  models = [None, None]
  for c in range(2):
    with tf.variable_scope(COLORS[c]):
      models[c] = model.Model(**chess_utils.MODEL_PARAMS)

  if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
    
  sess = tf.Session()
  saver = tf.train.Saver()

  with sess.as_default():
    writer = tf.train.SummaryWriter(args.model_dir, tf.get_default_graph())
    sess.run(tf.initialize_all_variables())

    model_path = os.path.join(args.model_dir, "model.ckpt")
    if os.path.isfile(model_path):
      saver.restore(sess, model_path)

    last_time = time.time()
    data_generator = ProcessData(GenerateData(models))
    for _ in range(args.num_train_steps):
      data = next(data_generator)
      for m, d, color in zip(models, data, COLORS):
        loss, global_step, _ = sess.run(
          [m.loss, m.global_step, m.optimize],
          feed_dict={
            m.observations: d.observations,
            m.actions: d.actions,
            m.rewards: d.rewards
          })

        if global_step % args.checkpoint_intervals == 0:
          saver.save(sess, model_path)
          cur_time = time.time()
          print("%s: loss at step %d is %f, took %.2f seconds." % 
            (color, global_step, loss, cur_time-last_time))
          last_time = cur_time


if __name__ == "__main__":
  Train()
