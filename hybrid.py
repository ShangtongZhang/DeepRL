import logging
from agent import *
from component import *
from utils import *
import argparse

class FruitHRFCNet(nn.Module, VanillaNet):
    def __init__(self, state_dim, action_dim, head_weights, optimizer_fn=None, gpu=True):
        super(FruitHRFCNet, self).__init__()
        hidden_size = 250
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.ModuleList([nn.Linear(hidden_size, action_dim) for _ in head_weights])
        self.criterion = nn.MSELoss()
        self.head_weights = head_weights
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x, heads_only):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        head_q = [fc(x) for fc in self.fc2]
        if not heads_only:
            q = [h * w for h, w in zip(head_q, self.head_weights)]
            q = torch.stack(q, dim=0)
            q = q.sum(0).squeeze(0)
            return q
        else:
            return head_q

    def predict(self, x, heads_only):
        return self.forward(x, heads_only)

class FruitMultiStatesFCNet(nn.Module, BasicNet):
    def __init__(self, state_dim, action_dim, head_weights, optimizer_fn=None, gpu=True):
        super(FruitMultiStatesFCNet, self).__init__()
        hidden_size = 250
        self.fc1 = nn.ModuleList([nn.Linear(state_dim, hidden_size) for _ in head_weights])
        self.fc2 = nn.ModuleList([nn.Linear(hidden_size, action_dim) for _ in head_weights])
        self.criterion = nn.MSELoss()
        self.head_weights = head_weights
        self.state_dim = state_dim
        self.n_heads = head_weights.shape[0]
        BasicNet.__init__(self, optimizer_fn, gpu)

    def predict(self, x, merge):
        head_q = []
        for i in range(self.n_heads):
            q = self.to_torch_variable(x[:, i, :])
            q = self.fc1[i](q)
            q = F.relu(q)
            q = self.fc2[i](q)
            head_q.append(q)
        if merge:
            q = [q * w for q, w in zip(head_q, self.head_weights)]
            q = torch.stack(q, dim=0)
            q = q.sum(0).squeeze(0)
            return q
        return head_q

FRUIT_EPISDOE_LENGTH = 100
MAX_EPISODES = 7000

class Fruit(BasicTask):
    def __init__(self, hybrid_reward=False, pseudo_reward=False, atomic_state=True):
        self.hybrid_reward = hybrid_reward
        self.atomic_state = atomic_state
        self.pseudo_reward = pseudo_reward
        self.name = "Fruit"
        self.success_threshold = 5
        self.width = 10
        self.height = 10
        self.possible_fruits = 10
        self.actual_fruits = 5
        xs = np.random.randint(0, self.width, size=self.possible_fruits)
        ys = np.random.randint(0, self.height, size=self.possible_fruits)
        self.possible_locations = list(zip(xs, ys))
        self.x = 0
        self.y = 0
        self.indices = np.arange(self.possible_fruits)
        self.taken = []
        self.remaining_fruits = 0

    def get_nearest(self):
        def distance(i):
            x, y = self.possible_locations[i]
            return np.abs(self.x - x) + np.abs(self.y - y)
        pool = []
        for i in range(self.possible_fruits):
            if not self.taken[i]:
                pool.append([i, distance(i)])
        pool = sorted(pool, key=lambda x:x[1])
        return pool[0][0]

    def encode_pos(self, x, y):
        return '{:04b}'.format(x) + '{:04b}'.format(y)

    def encode_atomic_state(self):
        offset = 8 * self.possible_fruits
        state = np.copy(self.base_state)
        str = self.encode_pos(self.x, self.y)
        for i in range(len(str)):
            state[offset + i] = int(str[i])
        offset += 8
        for i in range(len(self.taken)):
            state[offset + i] = self.taken[i]
        return state

    def encode_decomposed_state(self):
        state_size = (4 + 4) * 2 + 1
        base_state = np.zeros(state_size)
        str = self.encode_pos(self.x, self.y)
        for i in range(len(str)):
            base_state[i] = int(str[i])
        states = []
        for i in range(self.possible_fruits):
            states.append(np.copy(base_state))
            str = self.encode_pos(*self.possible_locations[i])
            for j in range(len(str)):
                states[-1][8 + j] = int(str[j])
            states[-1][-1] = self.taken[i]
        return np.asarray(states)

    def encode_state(self):
        if self.atomic_state:
            return self.encode_atomic_state()
        return self.encode_decomposed_state()

    def reset(self):
        self.x = np.random.randint(0, self.width)
        self.y = np.random.randint(0, self.height)
        np.random.shuffle(self.indices)
        self.taken = np.ones(self.possible_fruits, dtype=np.bool)
        self.taken[self.indices[: self.actual_fruits]] = False
        self.remaining_fruits = self.actual_fruits
        state_size = (4 + 4) * (self.possible_fruits + 1) + self.possible_fruits
        self.base_state = np.zeros(state_size)
        offset = 0
        for x, y in self.possible_locations:
            str = self.encode_pos(x, y)
            for i in range(len(str)):
                self.base_state[offset + i] = int(str[i])
            offset += 8
        return self.encode_state()

    def step(self, action):
        # action = action[0]
        if action == 0:
            self.x -= 1
        elif action == 1:
            self.x += 1
        elif action == 2:
            self.y -= 1
        elif action == 3:
            self.y += 1
        else:
            assert False
        self.x = min(max(self.x, 0), self.width - 1)
        self.y = min(max(self.y, 0), self.height - 1)
        try:
            pos = self.possible_locations.index((self.x, self.y))
        except ValueError:
            pos = -1
        if self.hybrid_reward:
            reward = np.zeros(self.possible_fruits)
            if pos >= 0 and not self.taken[pos]:
                reward[pos] = 1
                self.taken[pos] = True
                self.remaining_fruits -= 1
            if self.pseudo_reward:
                pseudo_reward = np.zeros(self.possible_fruits)
                if pos >= 0:
                    pseudo_reward[pos] = 1
                reward = (reward, pseudo_reward)
        else:
            reward = 0.0
            if pos >= 0 and not self.taken[pos]:
                reward = 1.0
                self.taken[pos] = True
                self.remaining_fruits -= 1
        return self.encode_state(), reward, not self.remaining_fruits, self.taken

BATCH_SIZE = 15

def dqn_fruit(args):
    config = Config()
    config.task_fn = lambda: Fruit()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, args.lr)
    # config.optimizer_fn = lambda params: torch.optim.SGD(params, lr)
    config.reward_weight = np.ones(10) / 10
    config.hybrid_reward = False
    config.network_fn = lambda optimizer_fn: FruitHRFCNet(
        98, 4, config.reward_weight, optimizer_fn)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=BATCH_SIZE)
    config.discount = 0.95
    config.target_network_update_freq = 200
    config.max_episode_length = FRUIT_EPISDOE_LENGTH
    config.exploration_steps = 200
    config.logger = Logger('./log', gym.logger)
    config.history_length = 1
    config.test_interval = 0
    config.test_repetitions = 10
    config.episode_limit = 5000
    config.tag = 'vanilla-%f' % (args.lr)
    config.double_q = False
    agent = DQNAgent(config)
    return agent

def hrdqn_fruit(args):
    config = Config()
    config.task_fn = lambda: Fruit(hybrid_reward=True)
    config.hybrid_reward = True
    config.reward_weight = np.ones(10) / 10
    config.optimizer_fn = lambda params: torch.optim.Adam(params, args.lr)
    # config.optimizer_fn = lambda params: torch.optim.SGD(params, 0.01)
    config.network_fn = lambda optimizer_fn: FruitHRFCNet(
        98, 4, config.reward_weight, optimizer_fn)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: HybridRewardReplay(memory_size=10000, batch_size=BATCH_SIZE)
    config.discount = 0.95
    config.target_network_update_freq = 200
    config.max_episode_length = FRUIT_EPISDOE_LENGTH
    config.exploration_steps = 200
    config.logger = Logger('./log', gym.logger)
    config.history_length = 1
    config.test_interval = 0
    config.test_repetitions = 10
    config.target_type = config.expected_sarsa_target
    config.tag = 'expected_sarsa-%f-%s' % (args.lr, args.tag)
    # config.target_type = config.q_target
    # config.tag = 'q-%f-%s' % (args.lr, args.tag)
    config.double_q = False
    config.episode_limit = 5000
    agent = DQNAgent(config)
    return agent

def hrmsdqn_fruit(args):
    config = Config()
    config.task_fn = lambda: Fruit(hybrid_reward=True, atomic_state=False)
    config.hybrid_reward = True
    # config.hybrid_reward = False
    config.reward_weight = np.ones(10) / 10
    # config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.optimizer_fn = lambda params: torch.optim.SGD(params, 0.1, momentum=0.9)
    config.network_fn = lambda optimizer_fn: FruitMultiStatesFCNet(
        17, 4, config.reward_weight, optimizer_fn)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: HybridRewardReplay(memory_size=10000, batch_size=BATCH_SIZE)
    config.discount = 0.95
    config.target_network_update_freq = 200
    config.max_episode_length = FRUIT_EPISDOE_LENGTH
    config.exploration_steps = 200
    config.logger = Logger('./log', gym.logger)
    config.history_length = 1
    config.test_interval = 0
    config.test_repetitions = 10
    config.target_type = config.expected_sarsa_target
    config.tag = 'expected_sarsa-%f-%s' % (args.lr, args.tag)
    # config.target_type = config.q_target
    # config.tag = 'q-%f-%s' % (args.lr, args.tag)
    config.double_q = False
    config.episode_limit = 5000
    agent = MSDQNAgent(config)
    return agent

if __name__ == '__main__':
    # gym.logger.setLevel(logging.DEBUG)
    gym.logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--tag', type=str, default='none')
    args = parser.parse_args()
    agent = hrdqn_fruit(args)
    # agent = hrmsdqn_fruit(args)
    agent.run()
