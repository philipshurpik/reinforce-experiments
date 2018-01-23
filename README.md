# reinforce-experiments
Simple implementations of vanilla reinforce (policy gradient) and actor critic methods with numpy and different frameworks

Current implementations:
PyTorch: 
 - based on: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
 - Actor Critic and Reinforce algorithms share the same code so it's easier to study these algorithms

Numpy:
 - based on Awesome Andrej Karpathy example: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
 - code was rewritten to make it simpler and to make it similar to pytorch examples
 - now works with different envs
 - works with different amount of actions - implemented softmax instead of sigmoid
 - actor critic version... in progress
 
Tensorflow:
 - ... in progress