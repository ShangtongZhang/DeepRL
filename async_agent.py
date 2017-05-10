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
import thread

class AsyncAgent:
    def __init__(self, task_fn, network_fn, optimizer_fn, policy_fn, discount, step_limit,
                 target_network_update_freq, n_workers):
        self.network_fn = network_fn
        self.learning_network = network_fn()
        self.learning_network.share_memory()
        self.target_network = network_fn()
        self.target_network.share_memory()
        self.target_network.load_state_dict(self.learning_network.state_dict())

        self.optimizer_fn = optimizer_fn
        # self.optimizer = optimizer_fn(self.learning_network.parameters())
        self.task_fn = task_fn
        self.step_limit = step_limit
        self.discount = discount
        self.target_network_update_freq = target_network_update_freq
        self.policy = policy_fn()
        self.total_steps = mp.Value('i', 0)
        self.lock = mp.Lock()
        self.n_workers = n_workers

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
        worker_network = self.network_fn()
        task = self.task_fn()
        episode = 0
        optimizer = self.optimizer_fn(self.learning_network.parameters())
        while True:
            worker_network.load_state_dict(self.learning_network.state_dict())
            worker_network.zero_grad()
            state = np.reshape(task.reset(), (1, -1))
            steps = 0
            total_reward = 0
            while not self.step_limit or steps < self.step_limit:
                value = worker_network.predict(state)
                action = self.policy.sample(value.flatten())
                next_state, reward, done, info = task.step(action)
                next_state = np.reshape(next_state, (1, -1))
                q_next = self.learning_network.predict(next_state)
                # q_next = self.target_network.predict(next_state)
                q_next = np.max(q_next, axis=1)
                if done:
                    q_next = 0
                q_next = reward + self.discount * q_next
                value[0, action] = q_next
                # if (steps > 0):
                #     print list(worker_network.parameters())[1].grad.data
                # worker_network.zero_grad()
                worker_network.gradient(state, value)
                # print list(worker_network.parameters())[1].grad
                # worker_network.zero_grad()
                # worker_network.gradient(state, value)
                # print list(worker_network.parameters())[1].grad
                # worker_network.gradient(state, value)
                # print list(worker_network.parameters())[1].grad

                steps += 1
                total_reward += reward
                with self.lock:
                    self.total_steps.value += 1
                    if self.total_steps.value % self.target_network_update_freq == 0:
                        self.target_network.load_state_dict(self.learning_network.state_dict())
                if done:
                    break
                state = next_state
            print 'worker %d, episode %d, rewards %d' % (id, episode, total_reward)
            episode += 1
            self.async_update(worker_network, optimizer)
            self.policy.update_epsilon()

    def run(self):
        procs = [mp.Process(target=self.worker, args=(i, )) for i in range(self.n_workers)]
        for p in procs: p.start()
        for p in procs: p.join()
        # print list(self.learning_network.parameters())[1].data

class Test:
    def __init__(self):
        # self.data = np.zeros((2, 3))
        self.data = torch.zeros((2, 3))
        self.data.share_memory_()
        print self.data
        self.val = mp.Value('i', 0)
        self.lock = mp.Lock()

    def fun(self):
        # self.data += rank
        for i in range(50):
            time.sleep(0.01)
            with self.lock:
                self.val.value += 1

    def run(self):
        processes = []
        for i in range(3):
            processes.append(mp.Process(target=self.fun))
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
    network_fn = lambda: FullyConnectedNet([4, 50, 200, 2])
    optimizer_fn = lambda params: torch.optim.SGD(params, 0.01)
    policy_fn = lambda: GreedyPolicy(epsilon=0.5, decay_factor=0.99, min_epsilon=0.01)
    # config = {'discount': 0.99, 'step_limit': 5000, 'target_network_update_freq': 200}
    agent = AsyncAgent(task_fn, network_fn, optimizer_fn, policy_fn, 0.99, 0, 200, 6)
    agent.run()

    # t = TestNet()
    # x = Variable(torch.from_numpy(np.array([[0.1, 0.2]], dtype='float32')))
    # target = Variable(torch.from_numpy(np.array([1], dtype='float32')))
    # t.gradient(x, target)
    # for p in t.parameters(): print p.grad
    # t.gradient(x, target)
    # for p in t.parameters(): print p.grad
    # t.gradient(x, target)
    # for p in t.parameters(): print p.grad
