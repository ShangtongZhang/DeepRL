#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *
from policy import *
import numpy as np
import torch.multiprocessing as mp
import time
from task import *
from network import *
from torch.autograd import Variable
import threading

class AsyncAgent:
    def __init__(self, task_fn, network_fn, optimizer_fn, policy_fn, discount, step_limit,
                 target_network_update_freq, n_workers):
        self.network_fn = network_fn
        self.learning_network = network_fn()
        # self.learning_network.share_memory()
        self.target_network = network_fn()
        # self.target_network.share_memory()
        self.target_network.load_state_dict(self.learning_network.state_dict())

        # self.optimizer_fn = optimizer_fn
        # self.optimizer = optimizer_fn(self.learning_network.parameters())
        self.task_fn = task_fn
        self.step_limit = step_limit
        self.discount = discount
        self.target_network_update_freq = target_network_update_freq
        # self.policy = policy_fn()
        self.policy_fn = policy_fn
        # self.total_steps = mp.Value('i', 0)
        self.total_steps = 0
        self.lock = threading.Lock()
        # self.lock = mp.Lock()
        self.n_workers = n_workers
        self.batch_size = 5

    def async_update(self, worker_network, optimizer):
        with self.lock:
            optimizer.zero_grad()
            for param, worker_param in zip(self.learning_network.parameters(), worker_network.parameters()):
                param._grad = worker_param.grad
            # print list(self.learning_network.parameters())[1].data
            # print list(worker_network.parameters())[1].grad.data
            optimizer.step()
            # print list(self.learning_network.parameters())[1].data

    def worker(self, id):
        # worker_network = self.network_fn()
        task = self.task_fn()
        episode = 0
        # optimizer = self.optimizer_fn(self.learning_network.parameters())
        policy = self.policy_fn()
        terminal = True
        episode_steps = 0
        while True:
            batch_states, batch_actions, batch_rewards = [], [], []
            if terminal:
                if id == 0:
                    print 'episode %d, epsilon %f, steps: %d' % \
                          (episode, policy.epsilon, episode_steps)
                episode_steps = 0
                episode += 1
                policy.update_epsilon()
                terminal = False
                state = task.reset()
                state = np.reshape(state, (1, -1))
            while not terminal and len(batch_states) < self.batch_size:
                episode_steps += 1
                with self.lock:
                    self.total_steps += 1
                batch_states.append(state)
                value = self.learning_network.predict(state)
                action = policy.sample(value.flatten())
                batch_actions.append(action)
                state, reward, terminal, _ = task.step(action)
                state = np.reshape(state, (1, -1))
                if not terminal:
                    q_next = np.max(self.target_network.predict(state))
                    reward += self.discount * q_next
                batch_rewards.append(reward)

            self.learning_network.learn_from_raw(np.vstack(batch_states), batch_actions, batch_rewards)

            if self.total_steps % self.target_network_update_freq == 0:
                with self.lock:
                    self.target_network.load_state_dict(self.learning_network.state_dict())

    def run(self):
        procs = [threading.Thread(target=self.worker, args=(i, )) for i in range(self.n_workers)]
        # procs = [mp.Process(target=self.worker, args=(i, )) for i in range(self.n_workers)]
        for p in procs: p.start()
        # while True:
        #     time.sleep(0.01)
        for p in procs: p.join()
        # print list(self.learning_network.parameters())[1].data

class Test:
    def __init__(self):
        # self.data = np.zeros((2, 3))
        self.data = torch.zeros((2, 3))
        # self.data.share_memory_()
        print self.data
        # self.val = mp.Value('i', 0)
        # self.lock = mp.Lock()
        self.lock = threading.Lock()

    def fun(self):
        # self.data += rank
        for i in range(50):
            time.sleep(0.01)
            # with self.lock:
            #     self.val.value += 1
            self.data += 1

    def run(self):
        processes = []
        for i in range(3):
            # processes.append(mp.Process(target=self.fun))
            processes.append(threading.Thread(target=self.fun))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc = nn.Linear(2, 1, bias=False)
        for param in self.parameters():
            param.data.copy_(torch.from_numpy(np.array([[1.0, 2.0]], dtype='float32').T))
        self.criterion = nn.MSELoss()
    def gradient(self, x, target):
        y = self.fc(x)
        loss = self.criterion(y, target)
        loss.backward()

# t = Test()
# t.run()
# print t.val.value
# print t.data

if __name__ == '__main__':
    task_fn = lambda: CartPole()
    optimizer_fn = lambda params: torch.optim.SGD(params, 0.001)
    network_fn = lambda: FullyConnectedNet([4, 50, 200, 2], optimizer_fn = optimizer_fn)
    policy_fn = lambda: GreedyPolicy(epsilon=1.0, end_episode=500, min_epsilon=0.1)
    # config = {'discount': 0.99, 'step_limit': 5000, 'target_network_update_freq': 200}
    agent = AsyncAgent(task_fn, network_fn, optimizer_fn, policy_fn, 0.99, 0, 200, 8)
    agent.run()

    # t = Test()
    # t.run()
    # print t.data

    # t = TestNet()
    # x = Variable(torch.from_numpy(np.array([[0.1, 0.2]], dtype='float32')))
    # target = Variable(torch.from_numpy(np.array([1], dtype='float32')))
    # t.gradient(x, target)
    # for p in t.parameters(): print p.grad
    # t.gradient(x, target)
    # for p in t.parameters(): print p.grad
    # t.gradient(x, target)
    # for p in t.parameters(): print p.grad
