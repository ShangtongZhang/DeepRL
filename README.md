# DeepRL
Highly modularized implementation of popular deep RL algorithms by PyTorch. My principal here is to
reuse as much components as I can through different algorithms, use as less tricks as I can and switch
easily between classical control tasks like CartPole and Atari games with raw pixel inputs.

Implemented algorithms:
* Deep Q-Learning (DQN)
* Double DQN
* Dueling DQN
* Async Advantage Actor Critic (A3C)
* Async One-Step Q-Learning
* Async One-Step Sarsa 
* Async N-Step Q-Learning
* Continuous A3C
* Deep Deterministic Policy Gradient (DDPG)
* Hybrid Reward Architecture (HRA)
* Distributed Proximal Policy Optimization (DPPO)

# Curves
> Curves for CartPole are trivial so I didn't place it here.
## DQN, Double DQN, Dueling DQN 
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/DQN-breakout.png)
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/DQN-Pong.png)

The network and parameters here are exactly same as the [DeepMind Nature paper](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). 
Training curve is smoothed by a window of size 100. All the models are trained in a server with
Xeon E5-2620 v3 and Titan X. For Breakout, test is triggered every 1000 episodes with 50 repetitions.
In total, 16M frames cost about 4 days and 10 hours. For Pong, test is triggered 
every 10 episodes with no repetition. In total, 4M frames cost about 18 hours.

## Discrete A3C

![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/A3C-Pong.png)
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/Async-Pong.png)

The network I used here is a smaller network with only 42 * 42 input, alougth the network for DQN can also work here,
it's quite slow. 

Training of A3C took about 2 hours (16 processes) in a server with two Xeon E5-2620 v3. While other async methods took about 1 day.
Those value based async methods do work but I don't know how to make them stable.
This is the test curve. Test is triggered in a separate deterministic test process every 50K frames.

## Continuous A3C
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/Continuous-A3C.png)

For continuous A3C and DPPO, I use fixed unit variance rather than a separate head, so entropy weight is simply set to 0.
Of course you can also use another head to output variance. In that case, a good practice is to bound your mean while leave 
variance unbounded, which is also included in the implementation.

## DDPG

## DPPO

The difference between my implementation and [DeepMind version](https://arxiv.org/abs/1707.02286) is:
1. PPO stands for different algorithms.
2. I use a much simpler A3C-like synchronization protocol. 

The body of PPO is based on [this](https://github.com/alexis-jacq/Pytorch-DPPO), however that implementation has some
 critical bugs. 

# Dependency
* Open AI gym
* PyTorch (For some reason I use v0.12 now, although I really like v0.2)
* Python 2.7 (I don't want to try Python 3 until I have to use RoboSchool)
* Tensorflow (Optional, but tensorboard is awesome)

# Usage
Detailed usage and all training parameters can be found in ```main.py```.
And you need to create following directories before running the program:
```
cd DeepRL
mkdir data log evaluation_log
```

# References
* [Human Level Control through Deep Reinforcement Learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730)
* [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
* [Hybrid Reward Architecture for Reinforcement Learning](https://arxiv.org/abs/1706.04208)
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)
* [transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)
* [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
* [ghliu/pytorch-ddpg](https://github.com/ghliu/pytorch-ddpg)
* [alexis-jacq/Pytorch-DPPO](https://github.com/alexis-jacq/Pytorch-DPPO)
