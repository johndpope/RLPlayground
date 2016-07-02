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
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--discard_draw", type=bool, default=False)
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
  "loss": "l2"
}


class Data(object):
  def __init__(self):
    self.observations = []
    self.actions = []
    self.rewards = []
    self.values = []


def GetObservations(games):
  observations = np.zeros([len(games), 64])
  for i, game in enumerate(games):
    board = game.GetState().board
    for j in xrange(64):
      observations[i, j] = board.At(j).Index()
  return observations


def SampleActions(games, values):
  actions = []
  sel_vals = []
  for i, game in enumerate(games):
    moves = game.GetMoves()
    vals = np.array([values[i, move.Index()] for move in moves])
    if np.random.binomial(1, args.epsilon, 1)[0]:
      idx = np.random.randint(len(moves))
    else:
      idx = np.argmax(vals)
    actions.append(moves[idx].Index())
    sel_vals.append(vals[idx])
  return np.array(actions, dtype=np.int32), np.array(sel_vals, dtype=np.float32) 


def PlayTurn(m, games):
  observations = GetObservations(games)
  values = m.outputs.eval(feed_dict={m.observations: observations})
  actions, values = SampleActions(games, values)
  rewards = np.zeros([len(games)], dtype=np.float32)
  for i, game in enumerate(games):
    if game.IsEnded():
      rewards[i] = 0
      continue
    r0 = chess.GetStateValue(game)
    game.Play(chess.Move(int(actions[i])))
    r1 = chess.GetStateValue(game)
    if game.IsEnded():
      rewards[i] = -r1
    else:
      rewards[i] = -r1 - r0
  return observations, actions, rewards, values


def UpdateData(data):
  # Convert to numpy
  observations = np.array(data.observations, dtype=np.int32)
  actions = np.array(data.actions, dtype=np.int32)
  rewards = np.array(data.rewards, dtype=np.float32)
  values = np.array(data.values, dtype=np.float32)

  # Compute updated value based on current value and the gained reward 
  new_values = rewards
  new_values[:-1] -= rewards[1:]
  new_values[:-2] += values[2:] * args.gamma
  
  # Update rewards and return
  data = Data()
  data.observations = observations
  data.actions = actions
  data.values = new_values
  return data


def GenerateData(models):
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
        turn = game.GetState().turn
        if game.IsCheckmate() or not args.discard_draw:
          if args.verbose:
            if game.IsCheckmate():
              print "%s won in %d steps." % (COLORS[turn], step[0])
            else:
              print "Draw in %d steps." % step[0]
          yield UpdateData(d)
        game.Reset()
        d.__init__()
        step[0] = 0
      else:
        step[0] += 1
    observations, actions, rewards, values = PlayTurn(models, games)
    for i, d in enumerate(data):
      d.observations.append(observations[i])
      d.actions.append(actions[i])
      d.rewards.append(rewards[i])
      d.values.append(values[i])


def MakeBatch(iterator):
  data = Data()
  for d in iterator:
    data.observations.append(d.observations)
    data.actions.append(d.actions)
    data.values.append(d.values)
    
    # Concatenate all the data
    if len(data.observations) == args.batch_size:
      cat_data = Data()      
      cat_data.observations = np.concatenate(data.observations)
      cat_data.actions = np.concatenate(data.actions)
      cat_data.values = np.concatenate(data.values)
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

    model_path = os.path.join(args.model_dir, "chess_qmodel.ckpt")
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
          m.targets: data.values
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
    model_path = os.path.join(args.model_dir, "chess_qmodel.ckpt")
    saver.restore(sess, model_path)
    while True:
      yield PlayTurn(m, [game])


if __name__ == "__main__":
  args = parser.parse_args()
  Train()
