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
parser.add_argument("--depth", type=int, nargs=2, default=[1, 1])
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


def GetObservation(game):
  observation = np.zeros([64])
  board = game.GetState().board
  for j in xrange(64):
    observation[j] = board.At(j).Index()
  return observation


def PlayTurn(game):
  observation = GetObservation(game)
  moves = game.GetMoves()
  turn = game.GetState().turn
  values = np.array(chess.GetActionValues(game, moves, args.depth[turn]))
  ind = np.random.choice(np.flatnonzero(values==values.max()))
  move = moves[ind]
  game.Play(move)
  return observation, move.Index()


def GenerateData():
  game = chess.Game()
  data = Data()
  step = 0

  # Play with current policy
  while True:
    if game.IsEnded() or step == args.max_game_steps:
      if args.verbose:
        if game.IsCheckmate():
          turn = game.GetState().turn
          print "%s won in %d steps." % (COLORS[turn], step)
        else:
          print "Draw in %d steps." % step
      yield data
      game.Reset()
      data.__init__()
      step = 0
    else:
      step += 1
    observation, action = PlayTurn(game)
    data.observations.append(observation)
    data.actions.append(action)


def MakeBatch(iterator):
  data = Data()
  for d in iterator:
    # Convert data to numpy and update the rewards
    data.observations.append(np.array(d.observations, dtype=np.int32))
    data.actions.append(np.array(d.actions, dtype=np.int32))
    
    # Concatenate all the data
    if len(data.observations) == args.batch_size:
      cat_data = Data()      
      cat_data.observations = np.concatenate(data.observations)
      cat_data.actions = np.concatenate(data.actions)
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
    data_generator = MakeBatch(GenerateData())
    for _ in xrange(args.num_train_steps):
      data = next(data_generator)
      loss, global_step, _ = sess.run(
        [m.loss, m.global_step, m.optimize],
        feed_dict={
          m.observations: data.observations,
          m.actions: data.actions,
          m.targets: np.ones(data.actions.shape, dtype=np.float32)
        })

      if global_step % args.checkpoint_intervals == 0:
        saver.save(sess, model_path)
        cur_time = time.time()
        print("Loss at step %d is %f, took %.2f seconds." % 
          (global_step, loss, cur_time-last_time))
        last_time = cur_time


if __name__ == "__main__":
  args = parser.parse_args()
  Train()
