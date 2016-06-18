import numpy as np
import random

import chess


MODEL_PARAMS = {
  "input_dim": 64, 
  "embedding_rows": 13, 
  "embedding_cols": 16, 
  "output_dim": 4096, 
  "hidden_dims": [2048, 2048, 2048], 
  "lr": 0.01, 
  "reg_factor": 0.0001
}


def SampleActions(games, ps):
  actions = []
  for i, game in enumerate(games):
    moves = game.GetMoves()
    if len(moves) > 0:
      probs = np.array([ps[i, move.Index()] for move in moves])
      idx = np.random.choice(probs.size, 1, p=probs/probs.sum())[0]
      actions.append(moves[idx].Index())
    else:
      actions.append(0)
  return np.array(actions, dtype=np.int32)


def GetObservations(games):
  observations = np.zeros([len(games), 64])
  for i, game in enumerate(games):
    board = game.GetState().board
    for j in range(64):
      observations[i, j] = board.At(j).Index()
  return observations


def PlayTurn(m, games, temperature=1):
  observations = GetObservations(games)
  ps = m.outputs.eval(feed_dict={
    m.observations: observations,
    m.temperature: temperature})
  actions = SampleActions(games, ps)
  for i, game in enumerate(games):
    game.Play(chess.Move(int(actions[i])))
  return observations, actions, np.zeros(actions.shape, dtype=np.float32)


def StandardValue(game):
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


def PlayRandom(game):
  moves = game.GetMoves()
  if len(moves) == 0:
    return
  sel = moves[random.randrange(len(moves))]
  game.Play(sel);


def PlayGreedy(game, depth=1, value_function=StandardValue):
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
