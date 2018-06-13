#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .atari_wrapper import *
import multiprocessing as mp
import sys
from .bench import Monitor
from ..utils import *
import uuid

class BaseTask:
    def set_monitor(self, env, log_dir):
        if log_dir is None:
            return env
        mkdir(log_dir)
        return Monitor(env, '%s/%s' % (log_dir, uuid.uuid4()))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if done:
            next_state = self.env.reset()
        return next_state, reward, done, info

    def seed(self, random_seed):
        return self.env.seed(random_seed)

class ClassicalControl(BaseTask):
    def __init__(self, name='CartPole-v0', max_steps=200, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

class PixelAtari(BaseTask):
    def __init__(self, name, seed=0, log_dir=None,
                 frame_skip=4, history_length=4, dataset=False, episode_life=True):
        BaseTask.__init__(self)
        env = make_atari(name, frame_skip)
        env.seed(seed)
        if dataset:
            env = DatasetEnv(env)
            self.dataset_env = env
        env = self.set_monitor(env, log_dir)
        env = wrap_deepmind(env, history_length=history_length, episode_life=episode_life)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape
        self.name = name

class RamAtari(BaseTask):
    def __init__(self, name, no_op, frame_skip, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        env = gym.make(name)
        assert 'NoFrameskip' in env.spec.id
        env = self.set_monitor(env, log_dir)
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=no_op)
        env = SkipEnv(env, skip=frame_skip)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = 128

class Pendulum(BaseTask):
    def __init__(self, log_dir=None):
        BaseTask.__init__(self)
        self.name = 'Pendulum-v0'
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(2 * action, -2, 2))

class Box2DContinuous(BaseTask):
    def __init__(self, name, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Roboschool(BaseTask):
    def __init__(self, name, log_dir=None):
        import roboschool
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Bullet(BaseTask):
    def __init__(self, name, log_dir=None):
        import pybullet_envs
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class PixelBullet(BaseTask):
    def __init__(self, name, seed=0, log_dir=None, frame_skip=4, history_length=4):
        import pybullet_envs
        self.name = name
        env = gym.make(name)
        env.seed(seed)
        env = RenderEnv(env)
        env = self.set_monitor(env, log_dir)
        env = SkipEnv(env, skip=frame_skip)
        env = WarpFrame(env)
        env = WrapPyTorch(env)
        if history_length:
            env = StackFrame(env, history_length)
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape
        self.env = env

class ProcessTask:
    def __init__(self, task_fn, log_dir=None):
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = ProcessWrapper(worker_pipe, task_fn, log_dir)
        self.worker.start()
        self.pipe.send([ProcessWrapper.SPECS, None])
        self.state_dim, self.action_dim, self.name = self.pipe.recv()

    def step(self, action):
        self.pipe.send([ProcessWrapper.STEP, action])
        return self.pipe.recv()

    def reset(self):
        self.pipe.send([ProcessWrapper.RESET, None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([ProcessWrapper.EXIT, None])

class ProcessWrapper(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    def __init__(self, pipe, task_fn, log_dir):
        mp.Process.__init__(self)
        self.pipe = pipe
        self.task_fn = task_fn
        self.log_dir = log_dir

    def run(self):
        np.random.seed()
        seed = np.random.randint(0, sys.maxsize)
        task = self.task_fn(log_dir=self.log_dir)
        task.seed(seed)
        while True:
            op, data = self.pipe.recv()
            if op == self.STEP:
                self.pipe.send(task.step(data))
            elif op == self.RESET:
                self.pipe.send(task.reset())
            elif op == self.EXIT:
                self.pipe.close()
                return
            elif op == self.SPECS:
                self.pipe.send([task.state_dim, task.action_dim, task.name])
            else:
                raise Exception('Unknown command')

class ParallelizedTask:
    def __init__(self, task_fn, num_workers, log_dir=None, single_process=False):
        if single_process:
            self.tasks = [task_fn(log_dir=log_dir) for _ in range(num_workers)]
        else:
            self.tasks = [ProcessTask(task_fn, log_dir) for _ in range(num_workers)]
        self.state_dim = self.tasks[0].state_dim
        self.action_dim = self.tasks[0].action_dim
        self.name = self.tasks[0].name
        self.single_process = single_process

    def step(self, actions):
        results = [task.step(action) for task, action in zip(self.tasks, actions)]
        results = map(lambda x: np.stack(x), zip(*results))
        return results

    def reset(self):
        results = [task.reset() for task in self.tasks]
        return np.stack(results)

    def close(self):
        if self.single_process:
            return
        for task in self.tasks: task.close()

class CliffWalking(gym.Env):
    POSITIVE_REWARD = 0
    NEGATIVE_REWARD = 1
    def __init__(self, random_action_prob, reward_type=NEGATIVE_REWARD, width=12, height=4):
        self.timeout = 100
        self.width = width
        self.height = height
        self.action_dim = 4
        self.S = (0, 0)
        self.G = (self.width - 1, 0)
        self.random_action_prob = random_action_prob
        self.reward_type = reward_type
        self.actions = [0, 1, 2, 3]
        self.action_space = gym.spaces.discrete.Discrete(4)
        self.observation_space = gym.spaces.box.Box(shape=(self.width * self.height, ), dtype=np.uint8,
                                                    low=0, high=1)

    def get_obs(self):
        obs = np.zeros(self.width * self.height, dtype=np.uint8)
        x, y = self.state
        obs[y * self.width + x] = 1
        return obs

    def fall(self):
        x, y = self.state
        return y == 0 and (0 < x < self.width - 1)

    def step(self, action):
        if np.random.rand() < self.random_action_prob:
            action = np.random.choice(self.actions)
        x, y = self.state
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.width - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.height - 1, y + 1)
        else:
            assert False, "Illegal Action"

        self.state = (x, y)
        self.steps += 1

        reward = -100 if self.fall() else -1
        done = True if self.fall() or self.state == self.G or self.steps == self.timeout else False

        return self.get_obs(), reward, done, {}

    def reset(self):
        self.state = self.S
        self.steps = 0
        return self.get_obs()

class CliffWalkingTask(BaseTask):
    def __init__(self, random_action_prob, log_dir=None, **kwargs):
        BaseTask.__init__(self)
        self.name = 'CliffWalking'
        self.env = CliffWalking(random_action_prob, **kwargs)
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

class IceCliffWalking(gym.Env):
    def __init__(self, random_action_prob=0, size=8, num_traps=8, penalty=-1):
        self.timeout = 100.0
        self.penalty = penalty
        self.size = size
        self.num_traps = num_traps
        self.random_action_prob = random_action_prob
        self.actions = [np.array([1, 0]),
                        np.array([0, -1]),
                        np.array([-1, 0]),
                        np.array([0, 1])]
        self.action_space = gym.spaces.discrete.Discrete(len(self.actions))
        self.observation_space = gym.spaces.box.Box(shape=(3, self.size, self.size), dtype=np.float32,
                                                    low=0, high=1)

    def rand_location(self):
        return np.random.randint(0, self.size, size=2).tolist()

    def dist(self, loc1, loc2):
        return np.abs(loc1[0] - loc2[0]) + np.abs(loc1[1] - loc2[1])

    def reset(self):
        self.agent = self.rand_location()
        self.goal = self.rand_location()
        while self.goal == self.agent or self.dist(self.agent, self.goal) <= 6:
            self.goal = self.rand_location()
        self.traps = []
        while len(self.traps) < self.num_traps:
            trap = self.rand_location()
            if trap != self.goal and trap != self.agent and (trap not in self.traps):
                self.traps.append(trap)
        self.steps = 0
        self.base_obs = np.zeros((4, self.size, self.size))
        self.base_obs[1, self.goal[0], self.goal[1]] = 1
        for trap in self.traps:
            self.base_obs[2, trap[0], trap[1]] = 1
        return self.get_obs()

    def get_obs(self):
        obs = np.copy(self.base_obs)
        obs[3] = self.steps / self.timeout
        obs[0, self.agent[0], self.agent[1]] = 1
        return obs

    def fall(self):
        x, y = self.agent
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        return False

    def slip(self):
        if self.agent in self.traps:
            return True
        return False

    def step(self, action):
        if np.random.rand() < self.random_action_prob:
            action = np.random.randint(0, len(self.actions))
        self.agent += self.actions[action]
        self.agent = self.agent.tolist()
        while self.slip():
            action = np.random.randint(0, len(self.actions))
            self.agent += self.actions[action]
            self.agent = self.agent.tolist()

        self.steps += 1

        if self.fall():
            reward = self.penalty
            done = True
        else:
            done = (self.agent == self.goal)
            reward = 10 if done else -0.1

        if done:
            next_obs = self.base_obs
        else:
            next_obs = self.get_obs()

        return next_obs, reward, done, {}

    def print(self):
        out = np.zeros([self.size, self.size])
        out[self.agent[0], self.agent[1]] = 1
        out[self.goal[0], self.goal[1]] = 2
        for trap in self.traps:
            out[trap[0], trap[1]] = 3
        chars = {0: ".", 1: "x", 2: "O", 3: "#"}
        pretty = "\n".join(["".join([chars[x] for x in row]) for row in out])
        print(pretty)
        print(self.steps)

class IceCliffWalkingTask(BaseTask):
    def __init__(self, log_dir=None, **kwargs):
        BaseTask.__init__(self)
        self.name = 'IceCliffWalking'
        self.env = IceCliffWalking(**kwargs)
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape
        self.env = self.set_monitor(self.env, log_dir)