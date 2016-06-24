# RL Playground #

RL Playground is a collection of environments and sample codes for experimenting
with reinforcement learning algorithms. The bulk of the source code at this
point is a chess environment. Tensorflow code for training policy gradient
algorithm is provided.

```
  a b c d e f g h
  ---------------
8 ◽ ◾ ◽ ◾ ◽ ◾ ◽ ◾ 8
7 ◾ ◽ ◾ ◽ ◾ ♝ ◾ ◽ 7
6 ◽ ♔ ◽ ◾ ◽ ◾ ♙ ♜ 6
5 ♗ ♟ ◾ ◽ ◾ ◽ ♖ ◽ 5
4 ♟ ◾ ♟ ◾ ♗ ◾ ◽ ◾ 4
3 ◾ ◽ ◾ ◽ ◾ ◽ ◾ ◽ 3
2 ◽ ♖ ◽ ◾ ◽ ◾ ◽ ◾ 2
1 ◾ ◽ ♜ ◽ ◾ ◽ ◾ ♚ 1
  ---------------
  a b c d e f g h
```


## Setup
Make sure _tensorflow_ and _swig_ are installed before building the 
package. Use _apt-get_ on Linux or _brew_ on Mac to install swig.
Follow instructions on [tensorflow's website](http//www.tensorflow.org)
to install tensorflow. 

To download and build the code run:
```
git clone https://github.com/vahidk/RLPlayground.git
cd RLPlayground
make
```

## Mutli-Armed Bandit Demo
This is the "hello world" example of reinforcement learning problems. The
environemnt is simply a slot machine with each slot returning a constant 
reward. The challenge is to maxmize total reward without knowing what the
optimal slot to select is.

To run the trainer run:
```
python py/mabp_train.py
```
After a few iterations you should see the model selecting the optimal
slot everytime. 

## Multiple Situations Demo
In this slightly more complicated example the agent is supposed to find
a diamond in a 1-D environemnt by selecting right or left actions. 
```
python py/diamond_train.py
``` 

## Chess demo
In this setup two independent agents are competing to win a chess game.
The code is work in progress.

To train the policy gradient model run:
```
python py/chess_pg_train.py
```

To train the Q-Learning model run:
```
python py/chess_q_train.py
```

Use the interactive demo to test out the environment:
``` 
python py/chess_interactive.py --mode random minimax
```
Note that you can specify the AI agent for black and white side separately.
Here we are setting the black agent to _random_, and the white agent to use
_minimax_ algorithm to select the move. You can also set this to _pg_ which
refers to the agent trained with policy gradient, or _q_ which uses Q-Learning
model to select moves.

At each step you can take control of the game by entering a move like "e4" or 
"e2e4".

Special commands:
- Enter: AI play one move.
- "x": AI play until the end.
- "r": Reset the board.
