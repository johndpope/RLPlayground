"""Auxilary functions for chess environment."""

import chess
import numpy as np

def GetStateValue(game):
  # Return the value of the current game state for the active player.
  if game.IsCheckmate():
    return -100
  elif game.IsDraw():
    return 0
  PIECE_VALUE = [0, 0, 9, 5, 3, 3, 1]
  state = game.GetState()
  pieces = state.board.GetPieces()
  multiplier = [-1, 1]
  value = 0.
  for piece in pieces:
    value += PIECE_VALUE[piece.type] * multiplier[state.turn==piece.color]
  return value


def GetActionValues(game, moves, depth=1, value_function=GetStateValue):
  turn = game.GetState().turn 
  values = []
  for move in moves:
    next_game = chess.Game(game)
    next_game.Play(move)
    value = -value_function(next_game)
    if depth > 1 and not next_game.IsEnded():
      next_moves = next_game.GetMoves()
      next_values = GetActionValues(
        next_game, next_moves, depth-1, value_function)
      value -= next_values.max()
    values.append(value)
  values = np.array(values, dtype=np.float)
  return values
