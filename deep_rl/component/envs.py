# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py

import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv as DummyVecEnv_
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

from ..utils import *

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        if is_atari:
            env = wrap_deepmind(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk


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

# Fix a bug in baselines, where the env cannot reset automatically
class DummyVecEnv(DummyVecEnv_):
    def __init__(self, env_fns):
        super(DummyVecEnv, self).__init__(env_fns)

    def step_wait(self):
        for i in range(self.num_envs):
            obs_tuple, self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = self.envs[i].step(self.actions[i])
            if self.buf_dones[i]:
                obs_tuple = self.envs[i].reset()
            if isinstance(obs_tuple, (tuple, list)):
                for t,x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return self.buf_obs[0], self.buf_rews, self.buf_dones, self.buf_infos

    def reset(self):
        obs = DummyVecEnv_.reset(self)
        return obs[0]

class Task:
    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs, reward, done, info


class VecTask(Task):
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, log_dir) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.env.name = name
        self.env.state_dim = self.env.observation_space.shape[0]
        self.env.action_dim = self.env.action_space.shape[0]

if __name__ == '__main__':
    task = VecTask('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
