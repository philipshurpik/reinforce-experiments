# reinforce-experiments
Simple implementations of vanilla reinforce (policy gradient) and actor critic methods with numpy, pytorch and tensorflow
By default works with CartPole and LunarLander

The main goal was to study algorithms and make how much possible amount of shared code between different implementations - to highlight differences between them 

Example commands to run:
```
python main.py --type numpy --model reinforce --env CartPole
python main.py --type pytorch --model reinforce --env CartPole
python main.py --type pytorch --model a2c --env CartPole
python main.py --type tensorflow --model reinforce --env LunarLander
```

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
 - based on numpy version
 - actor critic version... in progress