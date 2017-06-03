# DeepRL
Highly modularized implementation of popular deep RL algorithms by PyTorch. My principal here is to
reuse as much components as I can through different algorithms, use as less tricks as I can and switch
easily between classical control tasks like CartPole and Atari games with raw pixel inputs.

Implemented algorithms:
* Deep Q-Learning (DQN)
* Double DQN
* Dueling DQN
* Async One-Step Q-Learning
* Async One-Step Sarsa 
* Async N-Step Q-Learning
* Async Advantage Actor Critic (A3C)

# Curves
> Curves for CartPole is trivial so I didn't place it here.
## Deep Q-Learning (DQN)
![alt text](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/DQN-BreakoutNoFrameskip-v3-Train.png)
![alt_text](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/DQN-BreakoutNoFrameskip-v3-Test.png)

The network and parameters here are exactly same as the DeepMind Nature paper. 
Training curve is smoothed by window of size 100. Test is triggered every 1000 episodes.
In total it took about 16M frames. Training time is 4 days and 10 hours in a server with
Xeon E5-2620 v3 and Titan X.

## Asynchronous Advantage Actor Critic (A3C)

![alt text](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/A3C-PongNoFrameskip-v3.png)

The network I used here is same as the network in DQN except the activation function 
is **Elu** rather than Relu. The optimizer is **Adam** with non-shared parameters.
To my best knowledge, this network architecture is not the most suitable for A3C. 
If you use a 42 * 42 input, add a LSTM layer, you will get much much much better training speed 
than this. [GAE](http://www.breloff.com/DeepRL-OnlineGAE/) can also improve performance.
Another important thing is I didn't use lock for syncing up networks. Although I think there
should be a lock, locking can hurt the performance heavily (about 50%). 

The first 15 million frames took about 5 hours (16 processes) in a server with two Xeon E5-2620 v3.
This is the test curve. Test is triggered in a separate deterministic test process every 50K frames.

# Dependency
* Open AI gym
* PyTorch
* PIL (pip install Pillow)
* Python 2.7 (I didn't test with Python 3)

# Usage
Detailed usage and all training details can be found in ```main.py```

# References
* [Human Level Control through Deep Reinforcement Learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)
* [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
