import argparse
import numpy as np
import os
import time
import tensorflow as tf

import chess
import model

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model/")
parser.add_argument("--num_train_steps", type=int, default=1000000)
parser.add_argument("--checkpoint_intervals", type=int, default=1)
parser.add_argument("--max_game_steps", type=int, default=250)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--gamma", type=float, default=.99)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args([])

COLORS = ["Black", "White"]

MODEL_PARAMS = {
  "observations_dims": 64, 
  "observations_rows": 13, 
  "observations_cols": 16, 
  "actions_dims": 4096, 
  "hidden_dims": [1024] * 6, 
  "use_residual": True,
  "lr": args.learning_rate, 
  "reg_factor": 0.00001,
  "loss": "softmax"
}


class Data(object):
  def __init__(self):
    self.observations = []
    self.actions = []
    self.rewards = []


def GetObservations(games):
  observations = np.zeros([len(games), 64])
  for i, game in enumerate(games):
    board = game.GetState().board
    for j in xrange(64):
      observations[i, j] = board.At(j).Index()
  return observations


def SampleActions(games, ps):
  actions = []
  for i, game in enumerate(games):
    moves = game.GetMoves()
    probs = np.array([ps[i, move.Index()] for move in moves])
    idx = np.random.choice(probs.size, 1, p=probs/probs.sum())[0]
    actions.append(moves[idx].Index())
  return np.array(actions, dtype=np.int32)


def PlayTurn(m, games):
  observations = GetObservations(games)
  ps = m.outputs.eval(feed_dict={
    m.observations: observations})
  actions = SampleActions(games, ps)
  rs = np.zeros([len(games)], dtype=np.float32)
  for i, game in enumerate(games):
    if game.IsEnded():
      rs[i] = 0
      continue
    r0 = chess.GetStateValue(game)
    game.Play(chess.Move(int(actions[i])))
    r1 = chess.GetStateValue(game)
    if game.IsEnded():
      rs[i] = -r1
    else:
      rs[i] = -r1 - r0
  return observations, actions, rs


def GenerateData(m):
  games = []
  data = []
  steps = []
  for _ in xrange(args.batch_size):
    games.append(chess.Game())
    data.append(Data())
    steps.append([0])

  # Play with current policy
  while True:
    for step, d, game in zip(steps, data, games):
      if game.IsEnded() or step[0] == args.max_game_steps:
        if args.verbose:
          if game.IsCheckmate():
            turn = game.GetState().turn
            print "%s won in %d steps." % (COLORS[turn], step[0])
          else:
            print "Draw in %d steps." % step[0]
        yield d
        game.Reset()
        d.__init__()
        step[0] = 0
      else:
        step[0] += 1
    observations, actions, rewards = PlayTurn(m, games)
    for i, d in enumerate(data):
      d.observations.append(observations[i])
      d.actions.append(actions[i])
      d.rewards.append(rewards[i])


def ComputeDiscountedRewards(rs):
  alpha = 0
  for i in reversed(range(rs.size)):
    rs[i] += args.gamma * alpha
    alpha = rs[i]
  return rs


def MakeBatch(iterator):
  data = Data()
  for d in iterator:
    # Convert data to numpy and update the rewards
    data.observations.append(np.array(d.observations, dtype=np.int32))
    data.actions.append(np.array(d.actions, dtype=np.int32))
    data.rewards.append(
      ComputeDiscountedRewards(np.array(d.rewards, dtype=np.float32)))
    
    # Concatenate all the data
    if len(data.observations) == args.batch_size:
      cat_data = Data()      
      cat_data.observations = np.concatenate(data.observations)
      cat_data.actions = np.concatenate(data.actions)
      cat_data.rewards = np.concatenate(data.rewards)
      data.__init__()
      yield cat_data 


def Train():
  m = model.Model(**MODEL_PARAMS)

  if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
    
  sess = tf.Session()
  saver = tf.train.Saver()

  with sess.as_default():
    writer = tf.train.SummaryWriter(args.model_dir, tf.get_default_graph())
    sess.run(tf.initialize_all_variables())

    model_path = os.path.join(args.model_dir, "chess_pgmodel.ckpt")
    if os.path.isfile(model_path):
      saver.restore(sess, model_path)

    last_time = time.time()
    data_generator = MakeBatch(GenerateData(m))
    for _ in xrange(args.num_train_steps):
      data = next(data_generator)
      loss, global_step, _ = sess.run(
        [m.loss, m.global_step, m.optimize],
        feed_dict={
          m.observations: data.observations,
          m.actions: data.actions,
          m.targets: data.rewards
        })

      if global_step % args.checkpoint_intervals == 0:
        saver.save(sess, model_path)
        cur_time = time.time()
        print("Loss at step %d is %f, took %.2f seconds." % 
          (global_step, loss, cur_time-last_time))
        last_time = cur_time


def PlayTurnIterator(game):
  m = model.Model(**MODEL_PARAMS)

  if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
    
  sess = tf.Session()
  saver = tf.train.Saver()
  with sess.as_default():
    sess.run(tf.initialize_all_variables())
    model_path = os.path.join(args.model_dir, "chess_pgmodel.ckpt")
    saver.restore(sess, model_path)
    while True:
      yield PlayTurn(m, [game])

if __name__ == "__main__":
  args = parser.parse_args()
  Train()
