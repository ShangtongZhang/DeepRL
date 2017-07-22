#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import gym
import sys
import numpy as np
from atari_wrapper import *

class BasicTask:
    def __init__(self):
        self.normalized_state = True

    def normalize_state(self, state):
        return state

    def reset(self):
        state = self.env.reset()
        if self.normalized_state:
            return self.normalize_state(state)
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if self.normalized_state:
            next_state = self.normalize_state(next_state)
        return next_state, np.sign(reward), done, info

    def random_action(self):
        return self.env.action_space.sample()

class MountainCar(BasicTask):
    name = 'MountainCar-v0'
    success_threshold = -110

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize

class CartPole(BasicTask):
    name = 'CartPole-v0'
    success_threshold = 195

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize

class LunarLander(BasicTask):
    name = 'LunarLander-v2'
    success_threshold = 200

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)

class PixelAtari(BasicTask):
    def __init__(self, name, no_op, frame_skip, normalized_state=True,
                 frame_size=84, success_threshold=1000):
        BasicTask.__init__(self)
        self.normalized_state = normalized_state
        self.name = name
        self.success_threshold = success_threshold
        env = gym.make(name)
        assert 'NoFrameskip' in env.spec.id
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=no_op)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ProcessFrame(env, frame_size)
        self.env = ClippedRewardsWrapper(env)

    def normalize_state(self, state):
        return np.asarray(state, dtype=np.float32) / 255.0

class Pendulum(BasicTask):
    name = 'Pendulum-v0'
    success_threshold = 200

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = 2 * np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

class BipedalWalker(BasicTask):
    name = 'BipedalWalker-v2'
    success_threshold = 2000

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
