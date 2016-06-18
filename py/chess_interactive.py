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

def Reset():
  count[0] = 0
  game.Reset()

def PlayTurn():
  count[0] += 1
  depth = args.depth[game.GetState().turn]
  if depth == 0:
    chess_utils.PlayRandom(game)
  else:
    chess_utils.PlayGreedy(game, depth=depth) 

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
