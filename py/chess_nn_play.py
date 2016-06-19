import os
import random
import sys
import tensorflow as tf

import chess
import chess_utils
import pg_model

model_path = "model/model.ckpt"
COLORS = ["black", "white"]
models = [None] * 2

for c in range(2):
  with tf.variable_scope(COLORS[c]):
    models[c] = pg_model.Model(**chess_utils.MODEL_PARAMS)

sess = tf.Session()
saver = tf.train.Saver()

with sess.as_default():
  sess.run(tf.initialize_all_variables())
  if not os.path.isfile(model_path):
    print "No checkpoint found at %s." % model_path
    sys.exit()

  saver.restore(sess, model_path)
  print "Restored from heckpoint."

  game = chess.Game()

  while True:    
    state = game.GetState()
    print unicode(state.board)
    if game.IsCheckmate():
      print "Checkmate."
      print "%s wins." % COLORS[1-state.turn]
    elif game.IsDraw():
      print "Draw."
    else:
      print "%s turn." % COLORS[state.turn]

    valid_moves = game.GetMoves()
    if valid_moves:
      print "There are %d valid moves." % len(valid_moves)
    s = raw_input("Enter your command: ")
    
    if s == "x":
      while not game.IsEnded():
        state = game.GetState() 
        if state.turn:
          chess_utils.PlayTurn(models[state.turn], [game])
        else:
          moves = game.GetMoves()
          sel = moves[random.randrange(len(moves))]
          game.Play(sel);
    elif s == "":
      chess_utils.PlayTurn(models[state.turn], [game])
    elif s== "r":
      game.Reset()
    else:
      print "Didn't recognize the command."
