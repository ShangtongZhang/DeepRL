import numpy as np

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
            action = self.act(state, deterministic=True)
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

    def act(self, state, deterministic=False):
        if not deterministic and np.random.rand() < self.epsilon:
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

def run_episodes(agent):
    ep = 0
    while True:
        online_rewards, eval_rewards = agent.episode()
        ep += 1
        print('episode %d, return %f, eval return %f' % (ep, online_rewards, eval_rewards))

if __name__ == '__main__':
    agent = QAgent(lambda :Chain(10))
    run_episodes(agent)

