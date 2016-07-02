import argparse
import numpy as np
import random
import time

import chess
import chess_utils

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, nargs=2, 
  default=["minimax", "minimax"])
parser.add_argument("--depth", type=int, nargs=2, default=[1, 1])
args = parser.parse_args()

COLORS = ["Black", "White"]

count = [0]
game = chess.Game()
ai_player = [None]


def PlayRandom(game):
  moves = game.GetMoves()
  if len(moves) == 0:
    return
  sel = moves[random.randrange(len(moves))]
  game.Play(sel);


def PlayGreedy(game, depth=1, value_function=chess_utils.GetStateValue): 
  moves = game.GetMoves()
  values = chess_utils.GetActionValues(game, moves)
  ind = np.random.choice(np.flatnonzero(values==values.max()))
  game.Play(moves[ind])
  return values[ind]


def PlayPGModel(game):
  import chess_pg_train
  if ai_player[0] == None:
    ai_player[0] = chess_pg_train.PlayTurnIterator(game)
  next(ai_player[0])


def PlayQModel(game):
  import chess_q_train
  if ai_player[0] == None:
    ai_player[0] = chess_q_train.PlayTurnIterator(game)
  next(ai_player[0])


def Reset():
  count[0] = 0
  game.Reset()


def PlayTurn():
  count[0] += 1
  turn = game.GetState().turn
  mode = args.mode[turn]
  if mode == "random":
    PlayRandom(game)
  elif mode == "minimax":
    PlayGreedy(game, depth=args.depth[turn]) 
  elif mode == "pg":
    PlayPGModel(game)
  elif mode == "q":
    PlayQModel(game)
  else:
    print "Invalid mode."


def Main():
  while True:
    state = game.GetState()
    print unicode(state.board)
    if game.IsCheckmate():
      print "Checkmate."
      print "%s wins in %d moves." % (COLORS[1-state.turn], count[0])
    elif game.IsDraw():
      print "Draw."
    else:
      print "%s turn." % COLORS[state.turn]

    valid_moves = game.GetMoves()
    if valid_moves:
      print "There are %d valid moves." % len(valid_moves)
    s = raw_input("Enter your move: ")

    if s == "x":
      while not game.IsEnded():
        PlayTurn()
    elif s == "":
      if not game.IsEnded():
        PlayTurn()
    elif s == "r":
      Reset()
    elif s == "q":
      break
    else:
      for move in chess.Parse(state, s):
        if move in valid_moves:
          game.Play(move)
          break


if __name__ == "__main__":
  Main()
