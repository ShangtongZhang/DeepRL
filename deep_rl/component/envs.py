#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import indices
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from .tiles3 import tiles, IHT

from ..utils import *

try:
    import roboschool
except ImportError:
    pass


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, seed, rank, episode_life=True, tile_coding=False):
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        if tile_coding:
            env = TileCodingWrapper(env)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        return


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None,
                 tile_coding=False,
                 ):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, episode_life, tile_coding=tile_coding) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


class BairdPrediction(gym.Env):
    DASHED = 0
    SOLID = 1

    def __init__(self):
        self.num_states = 7
        self.phi = np.eye(7, 7) * 2
        self.phi[-1, -1] = 1
        self.phi = np.concatenate([self.phi, np.ones((7, 1))], axis=1)
        self.phi[-1, -1] = 2

        self.action_space = Discrete(2)
        self.observation_space = Box(-10, 10, (self.phi.shape[1],))

        self.state = None

    def reset(self):
        self.state = np.random.randint(self.num_states)
        return self.phi[self.state]

    def step(self, action):
        if action == self.DASHED:
            self.state = np.random.randint(self.num_states - 1)
        elif action == self.SOLID:
            self.state = self.num_states - 1
        else:
            raise NotImplementedError
        return self.phi[self.state], 0, False, {}

    def act(self, prob=6.0/7, pi_dashed=None):
        if np.random.rand() < prob:
            action = self.DASHED
            mu_prob = prob
            pi_prob = pi_dashed
        else:
            action = self.SOLID
            mu_prob = 1 - prob
            pi_prob = 1 - pi_dashed
        return dict(action=action, mu_prob=mu_prob, pi_prob=pi_prob)


class BairdControl(BairdPrediction):
    def __init__(self):
        super(BairdControl, self).__init__()
        self.phi_solid = self.phi
        self.phi_dash = np.eye(7, 7)
        self.phi = np.concatenate([self.phi_solid, self.phi_dash], axis=1)

        self.action_space = Discrete(2)
        self.observation_space = Box(-10, 10, (self.phi.shape[1],))

    def expand_phi(self, phi):
        assert len(phi.shape) == 1
        phi = np.array([phi, phi])
        phi[0, :8] = 0
        phi[1, 8:] = 0
        return phi


class TileCodingWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.max_size = 1024
        self.num_tilings = 8 
        high = np.where(self.observation_space.high > 100, 5, self.observation_space.high)
        low = np.where(self.observation_space.low < -100,
        -5, self.observation_space.low)
        env._max_episode_steps = 1000
        self.scale_factor = 10.0 / (high - low) 
        self.observation_space = Box(
            low = 0, high=1, shape=(self.max_size, ), 
            dtype=self.observation_space.dtype)
        
    
    def tile(self, obs):
        indices = tiles(self.max_size, self.num_tilings, obs * self.scale_factor)
        obs = np.zeros((self.max_size, ))
        obs[indices] = 1
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.tile(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.tile(obs)


if __name__ == '__main__':
    # task = Task('Hopper-v2', 5, single_process=False)
    task = Task('CartPole-v2', tile_coding=True)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
