# DeepRL
Highly modularized implementation of popular deep RL algorithms by PyTorch. My principal here is to reuse as much components as possible through different algorithms and switch easily between classical control tasks like CartPole and Atari games with raw pixel inputs.

Implemented algorithms:
* (Double/Dueling) Deep Q-Learning (DQN)
* Categorical DQN (C51, Distributional DQN with KL Distance)
* Quantile Regression DQN (Distributional DQN with Wasserstein Distance)
* Synchronous Advantage Actor Critic (A2C)
* Synchronous N-Step Q-Learning
* Deep Deterministic Policy Gradient (DDPG, pixel & low-dim-state)
* (Continuous/Discrete) Synchronous Proximal Policy Optimization (PPO, pixel & low-dim-state)
* The Option-Critic Architecture (OC)
* Action Conditional Video Prediction

Asynchronous algorithms below are removed in current version but can be found in [v0.1](https://github.com/ShangtongZhang/DeepRL/releases/tag/v0.1).
* Async Advantage Actor Critic (A3C)
* Async One-Step Q-Learning
* Async One-Step Sarsa 
* Async N-Step Q-Learning
* Continuous A3C
* Distributed Deep Deterministic Policy Gradient (Distributed DDPG, aka D3PG)
* Parallelized Proximal Policy Optimization (P3O, similar to DPPO)

Support for PyTorch v0.3.x can be found in [v0.2](https://github.com/ShangtongZhang/DeepRL/releases/tag/v0.2). Note all the figures are generated via this version. After the upgrade to PyTorch v0.4.0, I have only tested the classical control tasks. 

# Dependency
* MacOS 10.12 or Ubuntu 16.04
* PyTorch v0.4.0
* Python 3.6, 3.5 or 2.7 (deprecated)
* Core dependencies: `pip install -e .`
* Optional: [Roboschool](https://github.com/openai/roboschool), [PyBullet](https://pypi.org/project/pybullet/)

# Usage

```examples.py``` contains examples for all the implemented algorithms

```Dockerfile``` contains an example environment (w/ pybullet, w/o roboschool, w/o GPU)

Please use this bibtex if you want to cite this repo
```
@misc{deeprl,
  author = {Shangtong, Zhang},
  title = {Modularized Implementation of Deep RL Algorithms in PyTorch},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/ShangtongZhang/DeepRL}},
}
```

# Curves
> Curves for CartPole are trivial so I didn't place it here, and there isn't any fixed random seed. The curves are generated in the same manner as OpenAI baselines (one run and smoothed by recent 100 episodes)
## DQN
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/dqn_pixel_atari-180407-01414.png)

## Categorical DQN
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/categorical_dqn_pixel_atari-180407-094006.png)

## Quantile Regression DQN
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/quantile_regression_dqn_pixel_atari-180407-01604.png)

## A2C 
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/a2c_pixel_atari-180407-92711.png)

## N-Step Q-Learning
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/n_step_dqn_pixel_atari-180408-001104.png)

## DDPG 
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/ddpg_continuous-180407-234141.png)
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/ddpg-pixel.png)

## PPO 
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/ppo_continuous-180408-002056.png)
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/ppo_pixel_atari-180410-235529.png)

## OC
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/option_critic_pixel_atari-180417-092617.png)

This is my synchronous option-critic implementation, not the original one.

## Action Conditional Video Prediction
![Loading...](https://raw.githubusercontent.com/ShangtongZhang/DeepRL/master/images/ACVP.png)

**Left**: One-step prediction **Right**: Ground truth

Prediction is sampled after 110K iterations, and I only implemented one-step training

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
* [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
* [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
* Some hyper-parameters are from [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), [OpenAI Baselines](https://github.com/openai/baselines) and [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)