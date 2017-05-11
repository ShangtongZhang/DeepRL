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

class AsyncAgent:
    def __init__(self, task_fn, network_fn, optimizer_fn, policy_fn, discount, step_limit,
                 target_network_update_freq, n_workers, batch_size, test_interval):
        self.network_fn = network_fn
        self.learning_network = network_fn()
        self.learning_network.share_memory()
        self.target_network = network_fn()
        self.target_network.share_memory()
        self.target_network.load_state_dict(self.learning_network.state_dict())

        self.optimizer_fn = optimizer_fn
        self.task_fn = task_fn
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
        self.batch_size = batch_size
        self.test_interval = test_interval

    def deterministic_episode(self, task):
        state = np.asarray([task.reset()])
        total_rewards = 0
        steps = 0
        while True and steps < self.step_limit:
            with self.network_lock:
                action_values = self.learning_network.predict(state)
            steps += 1
            action = np.argmax(action_values.flatten())
            state, reward, terminal, _ = task.step(action)
            total_rewards += reward
            if terminal:
                break
            state = state.reshape([1, -1])
        return total_rewards

    def async_update(self, worker_network, optimizer):
        with self.network_lock:
            optimizer.zero_grad()
            for param, worker_param in zip(self.learning_network.parameters(), worker_network.parameters()):
                param._grad = worker_param.grad.clone()
            optimizer.step()

    def worker(self, id):
        optimizer = self.optimizer_fn(self.learning_network.parameters())
        worker_network = self.network_fn()
        worker_network.load_state_dict(self.learning_network.state_dict())
        task = self.task_fn()
        policy = self.policy_fn()
        terminal = True
        episode = 0
        episode_steps = 0
        while True and not self.stop_signal.value:
            batch_states, batch_actions, batch_rewards = [], [], []
            if terminal:
                episode_steps = 0
                episode += 1
                policy.update_epsilon()
                terminal = False
                state = task.reset()
                state = state.reshape([1, -1])
            while not terminal and len(batch_states) < self.batch_size:
                episode_steps += 1
                with self.steps_lock:
                    self.total_steps.value += 1
                batch_states.append(state)
                value = worker_network.predict(state)
                action = policy.sample(value.flatten())
                batch_actions.append(action)
                state, reward, terminal, _ = task.step(action)
                state = state.reshape([1, -1])
                if not terminal:
                    with self.network_lock:
                        q_next = np.max(self.target_network.predict(state))
                    reward += self.discount * q_next
                batch_rewards.append(reward)

            worker_network.zero_grad()
            worker_network.gradient(np.vstack(batch_states), batch_actions, batch_rewards)
            self.async_update(worker_network, optimizer)
            worker_network.load_state_dict(self.learning_network.state_dict())

            if self.total_steps.value % self.target_network_update_freq == 0:
                with self.network_lock:
                    self.target_network.load_state_dict(self.learning_network.state_dict())

    def run(self):
        procs = [mp.Process(target=self.worker, args=(i, )) for i in range(self.n_workers)]
        for p in procs: p.start()
        task = self.task_fn()
        while True:
            if self.total_steps.value % self.test_interval == 0:
                test_repeats = 5
                rewards = np.zeros(test_repeats)
                for i in range(test_repeats):
                    rewards[i] = self.deterministic_episode(task)
                print 'total stpes: %d, test process epsidoe reward: %f' %\
                      (self.total_steps.value, np.mean(rewards))
                if np.mean(rewards) > task.success_threshold:
                    self.stop_signal.value = True
                    break
        for p in procs: p.join()
