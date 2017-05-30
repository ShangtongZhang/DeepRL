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
    def transfer_state(self, state):
        return state

    def normalize_state(self, state):
        return state

    def reset(self):
        state = self.env.reset()
        return self.transfer_state(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.transfer_state(next_state)
        return next_state, np.sign(reward), done, info

class MountainCar(BasicTask):
    name = 'MountainCar-v0'
    success_threshold = -110

    def __init__(self):
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize

class CartPole(BasicTask):
    name = 'CartPole-v0'
    success_threshold = 195

    def __init__(self):
        self.env = gym.make(self.name)

class LunarLander(BasicTask):
    name = 'LunarLander-v2'
    success_threshold = 200

    def __init__(self):
        self.env = gym.make(self.name)

class PixelAtari(BasicTask):
    success_threshold = 1000

    def __init__(self, name, no_op, frame_skip):
        self.name = name
        env = gym.make(name)
        assert 'NoFrameskip' in env.spec.id
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=no_op)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ProcessFrame84(env)
        self.env = ClippedRewardsWrapper(env)

    def normalize_state(self, state):
        return np.asarray(state, dtype=np.float32) / 255.0
