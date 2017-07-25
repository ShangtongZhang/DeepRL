#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *
from policy import *
import numpy as np
import torch.multiprocessing as mp
from task import *
from network import *
from worker import *
import pickle
import os
import time

class AsyncAgent:
    def __init__(self, config):
        self.config = config
        learning_network = config.network_fn()
        learning_network.share_memory()
        target_network = config.network_fn()
        target_network.share_memory()
        target_network.load_state_dict(learning_network.state_dict())

        self.task = config.task_fn()

        self.config.learning_network = learning_network
        self.config.target_network = target_network
        self.config.steps_lock = mp.Lock()
        self.config.network_lock = mp.Lock()
        self.config.total_steps = mp.Value('i', 0)
        self.config.stop_signal = mp.Value('i', False)

    def train(self, id):
        worker = self.config.worker(self.config)
        episode = 0
        rewards = []
        while not self.config.stop_signal.value:
            steps, reward = worker.episode()
            rewards.append(reward)
            if len(rewards) > 100: rewards.pop(0)
            self.config.logger.debug('worker %d, episode %d, return %f, avg return %f, episode steps %d, total steps %d' % (
                id, episode, rewards[-1], np.mean(rewards[-100:]), steps, self.config.total_steps.value))

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.config.learning_network.state_dict(), f)

    def evaluate(self, id):
        test_rewards = []
        test_points = []
        worker = self.config.worker(self.config)
        while True:
            steps = self.config.total_steps.value
            if steps % self.config.test_interval == 0:
                worker.worker_network.load_state_dict(self.config.learning_network.state_dict())
                self.save('data/%s-%s-model-%s.bin' % (
                    self.config.tag, self.config.worker.__name__, self.task.name))
                rewards = np.zeros(self.config.test_repetitions)
                for i in range(self.config.test_repetitions):
                    rewards[i] = worker.episode(deterministic=True)[1]
                self.config.logger.info('total steps: %d, averaged return per episode: %f(%f)' %\
                      (steps, np.mean(rewards), np.std(rewards) / np.sqrt(self.config.test_repetitions)))
                test_rewards.append(np.mean(rewards))
                test_points.append(steps)
                with open('data/%s-%s-statistics-%s.bin' % (
                    self.config.tag, self.config.worker.__name__, self.task.name
                ), 'wb') as f:
                    pickle.dump([test_points, test_rewards], f)
                if np.mean(rewards) > self.task.success_threshold:
                    self.config.stop_signal.value = True
                    break

    def run(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        procs = [mp.Process(target=self.train, args=(i, )) for i in range(self.config.num_workers)]
        procs.append(mp.Process(target=self.evaluate, args=(self.config.num_workers, )))
        for p in procs: p.start()
        while True:
            time.sleep(1)
            for i, p in enumerate(procs):
                if not p.is_alive() and not self.config.stop_signal.value:
                    self.config.logger.warning('Worker %d exited unexpectedly.' % i)
                    p.terminate()
                    procs[i] = mp.Process(target=self.train, args=(i, ))
                    procs[i].start()
                    self.config.logger.warning('Worker %d restarted.' % i)
                    break
            if self.config.stop_signal.value:
                break
        for p in procs: p.join()
