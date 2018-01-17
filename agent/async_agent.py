#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch.multiprocessing as mp
from network import *
from utils import *
from component import *
from async_worker import *
import pickle
import os
import time
import sys

def train(id, config, learning_network, extra):
    np.random.seed()
    torch.manual_seed(np.random.randint(sys.maxsize))
    worker = config.worker(config, learning_network, extra)
    episode = 0
    rewards = []
    while not config.stop_signal.value:
        steps, reward = worker.episode()
        rewards.append(reward)
        if len(rewards) > 100: rewards.pop(0)
        config.logger.debug('worker %d, episode %d, return %f, avg return %f, episode steps %d, total steps %d' % (
            id, episode, rewards[-1], np.mean(rewards[-100:]), steps, config.total_steps.value))
        episode += 1

def evaluate(config, task, learning_network, extra):
    np.random.seed()
    torch.manual_seed(np.random.randint(sys.maxsize))
    test_rewards = []
    test_points = []
    test_wall_times = []
    initial_time = time.time()
    worker = config.worker(config, learning_network, extra)
    while True:
        steps = config.total_steps.value
        if config.test_interval and steps % config.test_interval == 0:
            worker.worker_network.load_state_dict(learning_network.state_dict())
            with open('data/%s-%s-model-%s.bin' % (
                    config.tag, config.worker.__name__, task.name), 'wb') as f:
                pickle.dump(learning_network.state_dict(), f)
            rewards = np.zeros(config.test_repetitions)
            for i in range(config.test_repetitions):
                rewards[i] = worker.episode(deterministic=True)[1]
            config.logger.info('total steps: %d, averaged return per episode: %f(%f)' % \
                               (steps, np.mean(rewards), np.std(rewards) / np.sqrt(config.test_repetitions)))
            test_rewards.append(np.mean(rewards))
            test_points.append(steps)
            test_wall_times.append(time.time() - initial_time)
            with open('data/%s-%s-statistics-%s.bin' % (
                    config.tag, config.worker.__name__, task.name), 'wb') as f:
                pickle.dump([test_rewards, test_points, test_wall_times], f)
            if np.mean(rewards) >= config.success_threshold or (config.max_steps and steps >= config.max_steps):
                config.stop_signal.value = True
                break

class AsyncAgent:
    def __init__(self, config):
        self.config = config
        self.config.steps_lock = mp.Lock()
        self.config.network_lock = mp.Lock()
        self.config.total_steps = mp.Value('i', 0)
        self.config.stop_signal = mp.Value('i', False)

    def run(self):
        config = self.config
        task = config.task_fn()
        learning_network = config.network_fn()
        learning_network.share_memory()

        os.environ['OMP_NUM_THREADS'] = '1'
        if config.worker == NStepQLearning or config.worker == OneStepQLearning or config.worker == OneStepSarsa:
            target_network = config.network_fn()
            target_network.share_memory()
            target_network.load_state_dict(learning_network.state_dict())
            extra = target_network
        elif config.worker == ContinuousAdvantageActorCritic \
                or config.worker == ProximalPolicyOptimization\
                or config.worker == DeterministicPolicyGradient:
            state_normalizer = StaticNormalizer(task.state_dim)
            reward_normalizer = StaticNormalizer(1)
            extra = [state_normalizer, reward_normalizer]
            if config.worker == DeterministicPolicyGradient:
                extra.append(config.replay_fn())
        else:
            extra = None
        args = [(i, config, learning_network, extra) for i in range(config.num_workers)]
        args.append((config, task, learning_network, extra))
        procs = [mp.Process(target=train, args=args[i]) for i in range(config.num_workers)]
        procs.append(mp.Process(target=evaluate, args=args[-1]))
        for p in procs: p.start()
        while True:
            time.sleep(1)
            for i, p in enumerate(procs):
                if not p.is_alive() and not config.stop_signal.value:
                    config.logger.warning('Worker %d exited unexpectedly.' % i)
                    p.terminate()
                    if i == config.num_workers:
                        target = evaluate
                    else:
                        target = train
                    procs[i] = mp.Process(target=target, args=args[i])
                    procs[i].start()
                    self.config.logger.warning('Worker %d restarted.' % i)
                    break
            if config.stop_signal.value:
                break
        for p in procs: p.join()
