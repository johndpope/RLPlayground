"""Auxilary functions for chess environment."""

def StandardValue(game):
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
