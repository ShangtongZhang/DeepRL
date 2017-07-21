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
import traceback
import time

class AsyncAgent:
    def __init__(self,
                 task_fn,
                 network_fn,
                 optimizer_fn,
                 policy_fn,
                 worker_fn,
                 discount,
                 step_limit,
                 target_network_update_freq,
                 n_workers,
                 update_interval,
                 test_interval,
                 test_repetitions,
                 history_length,
                 tag,
                 logger):
        self.network_fn = network_fn
        self.learning_network = network_fn()
        self.learning_network.share_memory()
        self.target_network = network_fn()
        self.target_network.share_memory()
        self.target_network.load_state_dict(self.learning_network.state_dict())
        self.worker_fn = worker_fn

        self.optimizer_fn = optimizer_fn
        self.task_fn = task_fn
        self.task = self.task_fn()
        self.step_limit = step_limit
        self.discount = discount
        self.optimizer_fn = optimizer_fn
        self.target_network_update_freq = target_network_update_freq
        self.policy_fn = policy_fn
        self.steps_lock = mp.Lock()
        self.network_lock = mp.Lock()
        self.total_steps = mp.Value('i', 0)
        self.stop_signal = mp.Value('i', False)
        self.n_workers = n_workers
        self.update_interval = update_interval
        self.test_interval = test_interval
        self.test_repetitions = test_repetitions
        self.logger = logger
        self.history_length = history_length
        self.tag = tag

    def deterministic_episode(self, task, network):
        state = task.reset()
        total_rewards = 0
        steps = 0
        network.reset(True)
        while not self.step_limit or steps < self.step_limit:
            action_value = network.predict(np.stack([state]))
            if self.worker_fn == AdvantageActorCritic:
                action_value = action_value[0]
            action = np.argmax(action_value.data.numpy().flatten())
            state, reward, terminal, _ = task.step(action)
            steps += 1
            total_rewards += reward
            if terminal:
                break
        return total_rewards

    def train(self, id):
        worker = self.worker_fn(self)
        episode = 0
        rewards = []
        while True and not self.stop_signal.value:
            steps, reward = worker.episode()
            rewards.append(reward)
            if len(rewards) > 100: rewards.pop(0)
            self.logger.debug('worker %d, episode %d, return %f, avg return %f, episode steps %d, total steps %d' % (
                id, episode, rewards[-1], np.mean(rewards[-100:]), steps, self.total_steps.value))

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.learning_network.state_dict(), f)

    def evaluate(self, id):
        test_rewards = []
        test_points = []
        test_network = self.network_fn()
        while True:
            steps = self.total_steps.value
            if steps % self.test_interval == 0:
                test_network.load_state_dict(self.learning_network.state_dict())
                self.save('data/%s%s-model-%s.bin' % (self.tag, self.worker_fn.__name__, self.task.name))
                rewards = np.zeros(self.test_repetitions)
                for i in range(self.test_repetitions):
                    rewards[i] = self.deterministic_episode(self.task, test_network)
                self.logger.info('total steps: %d, averaged return per episode: %f(%f)' %\
                      (steps, np.mean(rewards), np.std(rewards) / np.sqrt(self.test_repetitions)))
                test_rewards.append(np.mean(rewards))
                test_points.append(steps)
                with open('data/%s%s-statistics-%s.bin' % (
                    self.tag, self.worker_fn.__name__, self.task.name
                ), 'wb') as f:
                    pickle.dump([test_points, test_rewards], f)
                if np.mean(rewards) > self.task.success_threshold:
                    self.stop_signal.value = True
                    break

    def run(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        procs = [mp.Process(target=self.train, args=(i, )) for i in range(self.n_workers)]
        procs.append(mp.Process(target=self.evaluate, args=(self.n_workers, )))
        for p in procs: p.start()
        while True:
            time.sleep(1)
            for i, p in enumerate(procs):
                if not p.is_alive() and not self.stop_signal.value:
                    self.logger.warning('Worker %d exited unexpectedly.' % i)
                    p.terminate()
                    procs[i] = mp.Process(target=self.train, args=(i, ))
                    procs[i].start()
                    self.logger.warning('Worker %d restarted.' % i)
                    break
            if self.stop_signal.value:
                break
        for p in procs: p.join()
