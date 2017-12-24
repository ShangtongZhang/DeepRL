# This file is copied/apdated from
# https://raw.githubusercontent.com/transedward/pytorch-dqn/master/utils/atari_wrapper.py

import numpy as np
from collections import deque
import gym
from gym import spaces
from PIL import Image

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done  = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

def _process_frame84_rgb(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = Image.fromarray(img)
    resized_screen = img.resize((84, 110, 3), Image.BILINEAR)
    resized_screen = np.array(resized_screen)
    x_t = resized_screen[18:102, :, :]
    x_t = x_t.reshape((84, 84, 3))
    return x_t

class DatasetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(DatasetEnv, self).__init__(env)
        self.saved_obs = []
        self.saved_actions = []

    def get_saved(self):
        return self.saved_obs, self.saved_actions

    def clear_saved(self):
        self.saved_obs = []
        self.saved_actions = []

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.saved_actions.append(action)
        self.saved_obs.append(obs)
        return obs, reward, done, info

    def _reset(self):
        obs = self.env.reset()
        self.saved_obs.append(obs)
        return obs

def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    img = Image.fromarray(img)
    resized_screen = img.resize((84, 110), Image.BILINEAR)
    resized_screen = np.array(resized_screen)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [1, 84, 84])
    return x_t.astype(np.uint8)

def _process_frame42(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    img = img[34:34 + 160, :160]
    img = Image.fromarray(img)
    img = img.resize((80, 80), Image.BILINEAR)
    img = img.resize((42, 42), Image.BILINEAR)
    resized_screen = np.array(img).reshape(1, 42, 42)
    return resized_screen.astype(np.uint8)

class ProcessFrame(gym.Wrapper):
    def __init__(self, env=None, frame_size=84):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, frame_size, frame_size))
        if frame_size == 84:
            self.process_fn = _process_frame84
        elif frame_size == 42:
            self.process_fn = _process_frame42
        else:
            assert False, "Unknown frame size"

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.process_fn(obs), reward, done, info

    def _reset(self):
        return self.process_fn(self.env.reset())

class NormalizeFrame(gym.Wrapper):
    def __init__(self, env=None):
        super(NormalizeFrame, self).__init__(env)

    def _normalize(self, obs):
        return np.asarray(obs, dtype=np.float32) / 255.0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._normalize(obs), reward, done, info

    def _reset(self):
        return self._normalize(self.env.reset())
