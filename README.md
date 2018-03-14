# DeepRL
Highly modularized implementation of popular deep RL algorithms by PyTorch. My principal here is to
reuse as much components as I can through different algorithms, use as less tricks as I can and switch
easily between classical control tasks like CartPole and Atari games with raw pixel inputs.

Implemented algorithms:
* Deep Q-Learning (DQN)
* Double DQN
* Dueling DQN
* (Async) Advantage Actor Critic (A3C / A2C)
* Async One-Step Q-Learning
* Async One-Step Sarsa 
* Async N-Step Q-Learning
* Continuous A3C
* Distributed Deep Deterministic Policy Gradient (Distributed DDPG, aka D3PG)
* Parallelized Proximal Policy Optimization (P3O, similar to DPPO)
* Action Conditional Video Prediction
* Categorical DQN (C51, Distributional DQN)
* N-Step DQN (similar to A2C)

# Curves
> Curves for CartPole are trivial so I didn't place it here. There isn't any fixed random seed.
## DQN, Double DQN, Dueling DQN 
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/DQN-breakout.png)
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/DQN-Pong.png)

The network and parameters here are exactly same as the [DeepMind Nature paper](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). 
Training curve is smoothed by a window of size 100. All the models are trained in a server with
Xeon E5-2620 v3 and Titan X. For Breakout, test is triggered every 1000 episodes with 50 repetitions.
In total, 16M frames cost about 4 days and 10 hours. For Pong, test is triggered 
every 10 episodes with no repetition. In total, 4M frames cost about 18 hours.

I referred this [repo](https://github.com/transedward/pytorch-dqn).

## Discrete A3C

![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/A3C-Pong.png)
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/Async-Pong.png)

The network I used here is a smaller network with only 42 * 42 input, alougth the network for DQN can also work here,
it's quite slow. 

Training of A3C took about 2 hours (16 processes) in a server with two Xeon E5-2620 v3. While other async methods took about 1 day.
Those value based async methods do work but I don't know how to make them stable.
This is the test curve. Test is triggered in a separate deterministic test process every 50K frames.

I referred this [repo](https://github.com/ikostrikov/pytorch-a3c) for the parallelization.

## Continuous A3C
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/Continuous-A3C.png)

For continuous A3C and DPPO, I use fixed unit variance rather than a separate head, so entropy weight is simply set to 0.
Of course you can also use another head to output variance. In that case, a good practice is to bound your mean while leave 
variance unbounded, which is also included in the implementation.

## D3PG 

![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/DDPG.png)

Extra caution is necessary when computing gradients. The [repo](https://github.com/ghliu/pytorch-ddpg) I referred
for DDPG is wrong in computing the deterministic gradients at least at this [commit](https://github.com/ghliu/pytorch-ddpg/tree/ffea335ee53f2ff90b6d7eaf9d0cee705270c0f1).
Theoretically I believe that implementation should work, but in practice it doesn't work. Even this is PyTorch you need to manually deal with gradients in this case.
DDPG is not very stable. 

Setting the number of workers to 1 will reduce the implementation to exact DDPG. I have to adopt the most straightforward distribution method, as
P3O and A3C style distribution doesn't work for DDPG. The figures were done with 6 workers.


## P3O 

![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/P3O.png)

The difference between my implementation and [DeepMind's DPPO](https://arxiv.org/abs/1707.02286) is:
1. PPO stands for different algorithms.
2. I use a much simpler A3C-like synchronization protocol. 

The body of PPO is based on this [repo](https://github.com/alexis-jacq/Pytorch-DPPO). 
However that implementation has two critical bugs at least at this [commit](https://github.com/ghliu/pytorch-ddpg/tree/ffea335ee53f2ff90b6d7eaf9d0cee705270c0f1).
Its computation of the clipped loss is correct with one-dimensional action by accident, 
but is wrong with high-dimensional action. And its computation of entropy is wrong in any case.
 
I use 8 threads and a two tanh hidden layer network, each hidden layer has 64 hidden units.

## Action Conditional Video Prediction

![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/ACVP.png)

**Left**: One-step prediction **Right**: Ground truth

Prediction is sampled after 110K iterations and I only implemented one-step training

## Categorical DQN

![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/CategoricalDQN.png)
A deterministic test episode is triggered every 10 episodes. 2.5M steps and 14 hours in total.

## A2C & N-Step DQN
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/A2C-NStepQ.png)
Online training progression of a single run. Entropy regularization is used for A2C, resulting in the variance in the curve. 

# Dependency
> Tested in macOS 10.12 and CentO/S 6.8
* Open AI gym
* [Roboschool](https://github.com/openai/roboschool) (Optional)
* PyTorch v0.3.0
* Python 2.7 
* [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)


# Usage
```dataset.py```: generate dataset for action conditional video prediction

```main.py```: all other algorithms

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
* [Action-Conditional Video Prediction using Deep Networks in Atari Games](https://arxiv.org/abs/1507.08750)
* [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)