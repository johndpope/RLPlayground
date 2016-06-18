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
git pull https://github.com/vahidk/RLPlayground.git
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

To train the agents run:
```
python py/chess_train.py
```
This is probably going to take forever! Since the rewards at this point are
pretty simple. While Q-Learning algorithms are more suitable for this problem 
I implemented the policy gradient algorithm because of its flexibility.
Perhaps with better intermediate rewards and some supervised pre-training
it could be possible to make this work but at this point it doens't seem to
be doing much!    

Use the interactive demo to test out the environment:
``` 
python py/chess_interactive.py --depth 0 1
```
At each step you can enter a move like "e4" or "e2e4" to move a piece.

Special commands:
- Enter: Play one move.
- "x": Play until the end.
- "r": Reset the board.
