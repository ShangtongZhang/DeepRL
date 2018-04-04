#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import gym
import sys
import numpy as np
from .atari_wrapper import *
import multiprocessing as mp
import sys
from .bench import Monitor
from utils import *
import datetime
import uuid

class BasicTask:
    def __init__(self, max_steps=sys.maxsize):
        self.steps = 0
        self.max_steps = max_steps

    def reset(self):
        self.steps = 0
        state = self.env.reset()
        return state

    def normalize_state(self, state):
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.steps += 1
        done = (done or self.steps >= self.max_steps)
        return next_state, reward, done, info

    def random_action(self):
        return self.env.action_space.sample()

class ClassicalControl(BasicTask):
    def __init__(self, name='CartPole-v0', max_steps=200):
        BasicTask.__init__(self, max_steps)
        self.name = name
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]

class LunarLander(BasicTask):
    name = 'LunarLander-v2'
    success_threshold = 200

    def __init__(self, max_steps=sys.maxsize):
        BasicTask.__init__(self, max_steps)
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]

class PixelAtari(BasicTask):
    def __init__(self, name, seed=0, log_file=None, max_steps=sys.maxsize,
                 frame_skip=4, history_length=4):
        BasicTask.__init__(self, max_steps)
        env = make_atari(name, frame_skip)
        env.seed(seed)
        if log_file is None:
            log_dir = '%s-%s' % (
                name,
                datetime.datetime.now().strftime("%y%m%d-%-H%M%S"))
            mkdir('./log/%s' % log_dir)
            log_file = './log/%s/%s' % (log_dir, uuid.uuid1())
        env = Monitor(env, log_file)
        env = wrap_deepmind(env, history_length=history_length)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.name = name

    def normalize_state(self, state):
        return np.asarray(state) / 255.0

class RamAtari(BasicTask):
    def __init__(self, name, no_op, frame_skip, max_steps=10000):
        BasicTask.__init__(self, max_steps)
        self.name = name
        env = gym.make(name)
        assert 'NoFrameskip' in env.spec.id
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=no_op)
        env = SkipEnv(env, skip=frame_skip)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        self.env = env
        self.action_dim = self.env.action_space.n

    def normalize_state(self, state):
        return np.asarray(state) / 255.0

class ContinuousMountainCar(BasicTask):
    name = 'MountainCarContinuous-v0'
    success_threshold = 90

    def __init__(self, max_steps=sys.maxsize):
        BasicTask.__init__(self, max_steps)
        self.env = gym.make(self.name)
        self.max_episode_steps = self.env._max_episode_steps
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

class Pendulum(BasicTask):
    name = 'Pendulum-v0'
    success_threshold = -10

    def __init__(self, max_steps=sys.maxsize):
        BasicTask.__init__(self, max_steps)
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        return BasicTask.step(self, np.clip(action, -2, 2))

class Box2DContinuous(BasicTask):
    def __init__(self, name, max_steps=sys.maxsize):
        BasicTask.__init__(self, max_steps)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        return BasicTask.step(self, np.clip(action, -1, 1))

class Roboschool(BasicTask):
    def __init__(self, name, success_threshold=sys.maxsize, max_steps=sys.maxsize):
        import roboschool
        BasicTask.__init__(self, max_steps)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        return BasicTask.step(self, np.clip(action, -1, 1))

def sub_task(parent_pipe, pipe, task_fn, rank, log_dir):
    np.random.seed()
    seed = np.random.randint(0, sys.maxsize)
    parent_pipe.close()
    task = task_fn(log_file=os.path.join(log_dir, str(rank)))
    task.env.seed(seed)
    while True:
        op, data = pipe.recv()
        if op == 'step':
            pipe.send(task.step(data))
        elif op == 'reset':
            pipe.send(task.reset())
        elif op == 'exit':
            pipe.close()
            return
        else:
            assert False, 'Unknown Operation'

class ParallelizedTask:
    def __init__(self, task_fn, num_workers, tag='vanilla'):
        self.task_fn = task_fn
        self.task = task_fn()
        self.name = self.task.name
        log_dir = './log/%s-%s' % (self.name, tag)
        mkdir(log_dir)
        self.pipes, worker_pipes = zip(*[mp.Pipe() for _ in range(num_workers)])
        args = [(p, wp, task_fn, rank, log_dir)
                for rank, (p, wp) in enumerate(zip(self.pipes, worker_pipes))]
        self.workers = [mp.Process(target=sub_task, args=arg) for arg in args]
        for p in self.workers: p.start()
        for p in worker_pipes: p.close()
        self.observation_space = self.task.env.observation_space
        self.action_space = self.task.env.action_space

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        results = [p.recv() for p in self.pipes]
        results = map(lambda x: np.stack(x), zip(*results))
        return results

    def reset(self, i=None):
        if i is None:
            for pipe in self.pipes:
                pipe.send(('reset', None))
            results = [p.recv() for p in self.pipes]
        else:
            self.pipes[i].send(('reset', None))
            results = self.pipes[i].recv()
        return np.stack(results)

    def close(self):
        for pipe in self.pipes:
            pipe.send(('exit', None))
        for p in self.workers: p.join()

    def normalize_state(self, state):
        return self.task.normalize_state(state)
