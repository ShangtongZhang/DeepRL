import numpy as np
import torch

class Chain:
    def __init__(self, num_states, reward_std=0.1):
        self.num_states = num_states
        self.state = 0
        self.action_dim = 2
        self.state_dim = num_states
        self.reward_std = reward_std

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:
            self.state += 1
            reward = np.random.randn() * self.reward_std
            done = (self.state == self.num_states)
            if done:
                reward = 10.0
            return self.state, reward, done, None
        elif action == 1:
            return -1, 0, True, None

class BaseAgent:
    def eval(self):
        state = self.eval_env.reset()
        total_rewards = 0.0
        while True:
            action = self.act(state, eval=True)
            state, reward, done, _ = self.eval_env.step(action)
            total_rewards += reward
            if done:
                break
        return total_rewards

class QAgent(BaseAgent):
    def __init__(self, env_fn, lr=0.1, epsilon=0.1, discount=1.0):
        self.env = env_fn()
        self.eval_env = env_fn()
        self.q_values = np.zeros((self.env.state_dim, self.env.action_dim))
        self.epsilon = epsilon
        self.discount = discount
        self.lr = lr

    def act(self, state, eval=False):
        if not eval and np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_dim)
        else:
            best_q = np.max(self.q_values[state])
            candidates = [i for i in range(self.env.action_dim) if self.q_values[state, i] == best_q]
            return np.random.choice(candidates)

    def episode(self):
        state = self.env.reset()
        total_rewards = 0.0
        while True:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            total_rewards += reward

            if done:
                target = 0
            else:
                target = self.discount * np.max(self.q_values[next_state])
            td_error = reward + target - self.q_values[state, action]

            self.q_values[state, action] += self.lr * td_error

            if done:
                break
            state = next_state

        return total_rewards, self.eval()

class QuantileAgent(BaseAgent):
    def __init__(self, env_fn, lr=0.1, epsilon=0.1, discount=1.0, num_quantiles=5):
        self.env = env_fn()
        self.eval_env = env_fn()
        # self.q_values = np.zeros((self.env.state_dim, self.env.action_dim, num_quantiles))
        self.q_values = np.random.randn(*(self.env.state_dim, self.env.action_dim, num_quantiles))
        self.epsilon = epsilon
        self.discount = discount
        self.lr = lr
        self.num_quantiles = num_quantiles
        self.cumulative_density = (2 * np.arange(num_quantiles) + 1) / (2.0 * num_quantiles)
        self.cumulative_density = torch.FloatTensor(self.cumulative_density)

    def act(self, state, eval=False):
        if not eval:
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.env.action_dim)
            else:
                q = self.q_values[state].mean(-1)
                best_q = np.max(q)
                candidates = [i for i in range(self.env.action_dim) if q[i] == best_q]
                return np.random.choice(candidates)
        best_q = np.max(self.q_values[state, :, -1])
        candidates = [i for i in range(self.env.action_dim) if self.q_values[state, i, -1] == best_q]
        return np.random.choice(candidates)

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def episode(self):
        state = self.env.reset()
        total_rewards = 0.0
        while True:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            total_rewards += reward

            if done:
                target = np.zeros(self.num_quantiles)
            else:
                q_next = self.q_values[next_state, :].mean(-1)
                a_next = np.argmax(q_next)
                quantiles_next = self.q_values[next_state, a_next, :]
                target = self.discount * quantiles_next
            target += reward

            quantiles = self.q_values[state, action, :]

            quantiles_next = torch.FloatTensor(target).view(-1, 1)
            quantiles = torch.tensor(quantiles.astype(np.float32), requires_grad=True)
            diff = quantiles_next - quantiles
            loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()
            loss = loss.mean()
            loss.backward()

            self.q_values[state, action] -= self.lr * quantiles.grad.numpy().flatten()

            if done:
                break
            state = next_state

        print(self.q_values[:, 1, -1])
        return total_rewards, self.eval()

def run_episodes(agent):
    ep = 0
    while True:
        online_rewards, eval_rewards = agent.episode()
        ep += 1
        print('episode %d, return %f, eval return %f' % (ep, online_rewards, eval_rewards))

if __name__ == '__main__':
    # agent = QAgent(lambda :Chain(10))
    agent = QuantileAgent(lambda :Chain(8))
    run_episodes(agent)

