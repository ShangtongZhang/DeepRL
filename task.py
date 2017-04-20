import gym
import sys
from dqn_agent import *

class BasicTask:
    def transfer_state(self, state):
        return state

    def reset(self):
        return self.transfer_state(self.env.reset())

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.transfer_state(next_state)
        return next_state, reward, done, info

class MountainCar(BasicTask):
    state_space_size = 2
    action_space_size = 3
    name = 'MountainCar-v0'
    def __init__(self):
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize

if __name__ == '__main__':
    task = MountainCar()
    optimizer_fn = lambda name: tf.train.GradientDescentOptimizer(name=name, learning_rate=0.01)
    network_fn = lambda name: Network(name, task.state_space_size,
                                      task.action_space_size, optimizer_fn, tf.random_normal_initializer())
    policy_fn = lambda: GreedyPolicy(epsilon=0.5, decay_factor=0.95, min_epsilon=0.1)
    replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    agent = DQNAgent('mountain-car', task, network_fn, policy_fn, replay_fn,
                     discount=0.99, step_limit=5000, target_network_update_freq=1000)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ep = 0
        while True:
            ep += 1
            reward = agent.episode(sess)
            print 'episode %d: %f' % (ep, reward)
            if reward > -110:
                break
