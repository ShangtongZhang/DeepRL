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
from bootstrap import *
import pickle
import os

class AsyncAgent:
    def __init__(self,
                 task_fn,
                 network_fn,
                 optimizer_fn,
                 policy_fn,
                 bootstrap,
                 discount,
                 step_limit,
                 target_network_update_freq,
                 n_workers,
                 update_interval,
                 test_interval,
                 test_repetitions,
                 history_length,
                 logger):
        self.network_fn = network_fn
        self.learning_network = network_fn()
        self.learning_network.share_memory()
        if bootstrap != AdvantageActorCritic:
            self.target_network = network_fn()
            self.target_network.share_memory()
            self.target_network.load_state_dict(self.learning_network.state_dict())
        else:
            self.target_network = None
        self.bootstrap = bootstrap

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
        self.tag = ''

    def deterministic_episode(self, task, network):
        state = task.reset()
        total_rewards = 0
        steps = 0
        network.reset(True)
        bootstrap = self.bootstrap(self)
        while not self.step_limit or steps < self.step_limit:
            action = np.argmax(bootstrap.process_state(network, state))
            state, reward, terminal, _ = task.step(action)
            steps += 1
            total_rewards += reward
            if terminal:
                break
            bootstrap.reset()
        return total_rewards

    def worker(self, id):
        optimizer = self.optimizer_fn(self.learning_network.parameters())
        worker_network = self.network_fn()
        worker_network.load_state_dict(self.learning_network.state_dict())

        bootstrap = self.bootstrap(self)
        task = self.task_fn()
        policy = self.policy_fn()
        episode = 0
        episode_steps = 0
        episode_returns = [0]
        state = task.reset()
        pending_steps = 0
        while True and not self.stop_signal.value:
            action = policy.sample(bootstrap.process_state(worker_network, state))
            next_state, reward, terminal, _ = task.step(action)
            bootstrap.process_interaction(action, reward, next_state)

            episode_returns[-1] += reward
            episode_steps += 1
            if self.step_limit and episode_steps > self.step_limit:
                terminal = True
            with self.steps_lock:
                self.total_steps.value += 1
            pending_steps += 1

            if terminal or pending_steps >= self.update_interval:
                loss = bootstrap.compute_loss(worker_network, terminal)
                pending_steps = 0
                worker_network.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(worker_network.parameters(), 40)
                optimizer.zero_grad()
                for param, worker_param in zip(self.learning_network.parameters(), worker_network.parameters()):
                    param._grad = worker_param.grad.clone().cpu()
                optimizer.step()
                worker_network.load_state_dict(self.learning_network.state_dict())
                worker_network.reset(terminal)

            if terminal:
                state = task.reset()
                episode += 1
                if id == 0:
                    self.logger.info('episode %d, return %f, avg return %f, episode steps %d, total steps %d' % (
                        episode, episode_returns[-1], np.mean(episode_returns[-100:]), episode_steps, self.total_steps.value))
                episode_returns.append(0)
                episode_steps = 0
            else:
                state = next_state

            if self.target_network and self.total_steps.value % self.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.learning_network.state_dict(), f)

    def run(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        procs = [mp.Process(target=self.worker, args=(i, )) for i in range(self.n_workers)]
        for p in procs: p.start()
        test_rewards = []
        test_points = []
        test_network = self.network_fn()
        while True:
            steps = self.total_steps.value + 1
            if steps % self.test_interval == 0:
                test_network.load_state_dict(self.learning_network.state_dict())
                self.save('data/%s%s-model-%s.bin' % (self.tag, self.bootstrap.__name__, self.task.name))
                rewards = np.zeros(self.test_repetitions)
                for i in range(self.test_repetitions):
                    rewards[i] = self.deterministic_episode(self.task, test_network)
                self.logger.info('total steps: %d, averaged return per episode: %f(%f)' %\
                      (steps, np.mean(rewards), np.std(rewards) / np.sqrt(self.test_repetitions)))
                test_rewards.append(np.mean(rewards))
                test_points.append(steps)
                with open('data/%s%s-statistics-%s.bin' % (
                    self.tag, self.bootstrap.__name__, self.task.name
                ), 'wb') as f:
                    pickle.dump([test_points, test_rewards], f)
                if np.mean(rewards) > self.task.success_threshold:
                    self.stop_signal.value = True
                    break
        for p in procs: p.join()
