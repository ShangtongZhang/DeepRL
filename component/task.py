#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import gym
import sys
import numpy as np
from .atari_wrapper import *

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
        self.env = ProcessFrame(env, frame_size)
        self.action_dim = self.env.action_space.n

    def normalize_state(self, state):
        return np.asarray(state, dtype=np.float32) / 255.0

class ContinuousMountainCar(BasicTask):
    name = 'MountainCarContinuous-v0'
    success_threshold = 90

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.max_episode_steps = self.env._max_episode_steps
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info


class Pendulum(BasicTask):
    name = 'Pendulum-v0'
    success_threshold = -10

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.max_episode_steps = self.env._max_episode_steps
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -2, 2)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

class BipedalWalker(BasicTask):
    name = 'BipedalWalker-v2'
    success_threshold = 300

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.max_episode_steps = self.env._max_episode_steps
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

class BipedalWalkerHardcore(BasicTask):
    name = 'BipedalWalkerHardcore-v2'
    success_threshold = 300

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.max_episode_steps = self.env._max_episode_steps
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

class ContinuousLunarLander(BasicTask):
    name = 'LunarLanderContinuous-v2'
    success_threshold = 300

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.max_episode_steps = self.env._max_episode_steps
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

class Roboschool(BasicTask):
    def __init__(self, name, success_threshold=sys.maxsize, max_episode_steps=None):
        import roboschool
        BasicTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.success_threshold = success_threshold
        if max_episode_steps is None:
            self.max_episode_steps = self.env._max_episode_steps
        else:
            self.max_episode_steps = max_episode_steps
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

class Fruit(BasicTask):
    def __init__(self, hybrid_reward=False, pseudo_reward=False, atomic_state=True):
        self.hybrid_reward = hybrid_reward
        self.atomic_state = atomic_state
        self.pseudo_reward = pseudo_reward
        self.name = "Fruit"
        self.success_threshold = 5
        self.width = 10
        self.height = 10
        self.possible_fruits = 10
        self.actual_fruits = 5
        xs = np.random.randint(0, self.width, size=self.possible_fruits)
        ys = np.random.randint(0, self.height, size=self.possible_fruits)
        self.possible_locations = list(zip(xs, ys))
        self.x = 0
        self.y = 0
        self.indices = np.arange(self.possible_fruits)
        self.taken = []
        self.remaining_fruits = 0

    def get_nearest(self):
        def distance(i):
            x, y = self.possible_locations[i]
            return np.abs(self.x - x) + np.abs(self.y - y)
        pool = []
        for i in range(self.possible_fruits):
            if not self.taken[i]:
                pool.append([i, distance(i)])
        pool = sorted(pool, key=lambda x:x[1])
        return pool[0][0]

    def encode_pos(self, x, y):
        return '{:04b}'.format(x) + '{:04b}'.format(y)

    def encode_atomic_state(self):
        offset = 8 * self.possible_fruits
        state = np.copy(self.base_state)
        str = self.encode_pos(self.x, self.y)
        for i in range(len(str)):
            state[offset + i] = int(str[i])
        offset += 8
        for i in range(len(self.taken)):
            state[offset + i] = self.taken[i]
        return state

    def encode_decomposed_state(self):
        state_size = (4 + 4) * 2 + 1
        base_state = np.zeros(state_size)
        str = self.encode_pos(self.x, self.y)
        for i in range(len(str)):
            base_state[i] = int(str[i])
        states = []
        for i in range(self.possible_fruits):
            states.append(np.copy(base_state))
            str = self.encode_pos(*self.possible_locations[i])
            for j in range(len(str)):
                states[-1][8 + j] = int(str[j])
            states[-1][-1] = self.taken[i]
        return np.asarray(states)

    def encode_state(self):
        if self.atomic_state:
            return self.encode_atomic_state()
        return self.encode_decomposed_state()

    def reset(self):
        self.x = np.random.randint(0, self.width)
        self.y = np.random.randint(0, self.height)
        np.random.shuffle(self.indices)
        self.taken = np.ones(self.possible_fruits, dtype=np.bool)
        self.taken[self.indices[: self.actual_fruits]] = False
        self.remaining_fruits = self.actual_fruits
        state_size = (4 + 4) * (self.possible_fruits + 1) + self.possible_fruits
        self.base_state = np.zeros(state_size)
        offset = 0
        for x, y in self.possible_locations:
            str = self.encode_pos(x, y)
            for i in range(len(str)):
                self.base_state[offset + i] = int(str[i])
            offset += 8
        return self.encode_state()

    def step(self, action):
        # action = action[0]
        if action == 0:
            self.x -= 1
        elif action == 1:
            self.x += 1
        elif action == 2:
            self.y -= 1
        elif action == 3:
            self.y += 1
        else:
            assert False
        self.x = min(max(self.x, 0), self.width - 1)
        self.y = min(max(self.y, 0), self.height - 1)
        try:
            pos = self.possible_locations.index((self.x, self.y))
        except ValueError:
            pos = -1
        if self.hybrid_reward:
            reward = np.zeros(self.possible_fruits)
            if pos >= 0 and not self.taken[pos]:
                reward[pos] = 10
                self.taken[pos] = True
                self.remaining_fruits -= 1
            if self.pseudo_reward:
                pseudo_reward = np.zeros(self.possible_fruits)
                if pos >= 0:
                    pseudo_reward[pos] = 1
                reward = (reward, pseudo_reward)
        else:
            reward = 0.0
            if pos >= 0 and not self.taken[pos]:
                reward = 1.0
                self.taken[pos] = True
                self.remaining_fruits -= 1
        return self.encode_state(), reward, not self.remaining_fruits, self.taken