import argparse
import random
import time

import chess
import chess_utils

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=int, nargs=2, default=[1, 1])
args = parser.parse_args()

COLORS = ["Black", "White"]

count = [0]
game = chess.Game()


def PlayRandom(game):
  moves = game.GetMoves()
  if len(moves) == 0:
    return
  sel = moves[random.randrange(len(moves))]
  game.Play(sel);


def PlayGreedy(game, depth=1, value_function=chess_utils.StandardValue):
  turn = game.GetState().turn 
  moves = game.GetMoves()
  values = []
  for move in moves:
    new_game = chess.Game(game)
    new_game.Play(move)
    value = -value_function(new_game)
    if depth > 1 and not new_game.IsEnded():
      value -= PlayGreedy(new_game, depth-1, value_function)
    values.append(value)
  values = np.array(values, dtype=np.float)
  ind = np.random.choice(np.flatnonzero(values==values.max()))
  game.Play(moves[ind])
  return values[ind]


def Reset():
  count[0] = 0
  game.Reset()


def PlayTurn():
  count[0] += 1
  depth = args.depth[game.GetState().turn]
  if depth == 0:
    PlayRandom(game)
  else:
    PlayGreedy(game, depth=depth) 


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
