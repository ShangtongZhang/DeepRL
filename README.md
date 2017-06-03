# DeepRL
Highly modularized implementation of popular deep RL algorithms by PyTorch. My principal here is to
reuse as much components as I can through different algorithms and use as less tricks as I can.
* Deep Q-Learning
* Asynchronous One-Step Q-Learning
* Asynchronous One-Step Sarsa 
* Asynchronous N-Step Q-Learning
* Asynchronous Advantage Actor Critic (A3C)

>Tested with both classical control tasks (CartPole) and Atari games.

# Curves

## Asynchronous Advantage Actor Critic (A3C)

![alt text](DeepRL/images/A3C-PongNoFrameskip-v3.png)

The network I used here is same as the network in DQN except the activation function 
is **Elu** rather than Relu. The optimizer is **Adam** with non-shared parameters.
To my best knowledge, this network architecture is not the most suitable for A3C. 
If you use a 42 * 42 input, add a LSTM layer, you will get much much much better training speed 
than this. [GAE](http://www.breloff.com/DeepRL-OnlineGAE/) can also improve performance.
Another important thing is I didn't use lock for syncing up networks. Although I think there
should be a lock, locking can hurt the performance heavily (about 50%). 

The first 15 million frames took about 5 hours (16 processes) in a server with two Xeon E5-2620 v3.

# Usage
Detailed usage and all training details can be found in ```main.py```

# References
* [Human Level Control through Deep Reinforcement Learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)
* [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
