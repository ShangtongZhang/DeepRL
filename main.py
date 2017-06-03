from async_agent import *
from dqn_agent import *
import logging

def dqn_cart_pole():
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.RMSprop(params, 0.001)
    config['network_fn'] = lambda optimizer_fn: FullyConnectedNet([8, 50, 200, 2], optimizer_fn)
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config['replay_fn'] = lambda: Replay(memory_size=10000, batch_size=10)
    config['discount'] = 0.99
    config['target_network_update_freq'] = 200
    config['step_limit'] = 0
    config['explore_steps'] = 1000
    config['logger'] = gym.logger
    config['history_length'] = 2
    config['test_interval'] = 100
    config['test_repetitions'] = 50
    agent = DQNAgent(**config)
    agent.run()

def async_cart_pole():
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, 0.001)
    config['network_fn'] = lambda: FullyConnectedNet([4, 50, 200, 2], gpu=False)
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, final_step=5000, min_epsilon=0.1)
    config['bootstrap_fn'] = OneStepQLearning
    # config['bootstrap_fn'] = NStepQLearning
    # config['bootstrap_fn'] = OneStepSarsa
    config['discount'] = 0.99
    config['target_network_update_freq'] = 200
    config['step_limit'] = 0
    config['n_workers'] = 16
    config['batch_size'] = 6
    config['test_interval'] = 4000
    config['test_repetitions'] = 50
    config['history_length'] = 1
    config['logger'] = gym.logger
    agent = AsyncAgent(**config)
    agent.run()

def a3c_cart_pole():
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, 0.001)
    config['network_fn'] = lambda: FCActorCriticNet([4, 200, 2], gpu=False)
    config['policy_fn'] = SamplePolicy
    config['bootstrap_fn'] = AdvantageActorCritic
    config['discount'] = 0.99
    config['target_network_update_freq'] = 0
    config['step_limit'] = 0
    config['n_workers'] = 16
    config['batch_size'] = 6
    config['test_interval'] = 4000
    config['history_length'] = 1
    config['test_repetitions'] = 50
    config['logger'] = gym.logger
    agent = AsyncAgent(**config)
    agent.run()

def dqn_pixel_atari(name):
    config = dict()
    history_length = 4
    n_actions = 6
    config['task_fn'] = lambda: PixelAtari(name, no_op=30, frame_skip=4, normalized_state=False)
    config['optimizer_fn'] = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config['network_fn'] = lambda optimizer_fn: ConvNet(history_length, n_actions, optimizer_fn)
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config['replay_fn'] = lambda: Replay(memory_size=1000000, batch_size=32, dtype=np.uint8)
    config['discount'] = 0.99
    config['target_network_update_freq'] = 10000
    config['step_limit'] = 0
    config['explore_steps'] = 50000
    config['logger'] = gym.logger
    config['history_length'] = history_length
    config['test_interval'] = 1000
    config['test_repetitions'] = 50
    agent = DQNAgent(**config)
    agent.run()

def async_pixel_atari(name):
    config = dict()
    history_length = 1
    n_actions = 6
    config['task_fn'] = lambda: PixelAtari(name, no_op=30, frame_skip=4)
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, lr=0.0001)
    config['network_fn'] = lambda: ConvNet(history_length, n_actions, gpu=False)
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config['bootstrap_fn'] = OneStepQLearning
    # config['bootstrap_fn'] = NStepQLearning
    # config['bootstrap_fn'] = OneStepSarsa
    config['discount'] = 0.99
    config['target_network_update_freq'] = 10000
    config['step_limit'] = 10000
    config['n_workers'] = 16
    config['batch_size'] = 20
    config['test_interval'] = 50000
    config['test_repetitions'] = 1
    config['history_length'] = history_length
    config['logger'] = gym.logger
    agent = AsyncAgent(**config)
    agent.run()

def a3c_pixel_atari(name):
    config = dict()
    history_length = 1
    n_actions = 6
    config['task_fn'] = lambda: PixelAtari(name, no_op=30, frame_skip=4)
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, lr=0.0001)
    config['network_fn'] = lambda: ConvActorCriticNet(history_length, n_actions, gpu=False)
    config['policy_fn'] = SamplePolicy
    config['bootstrap_fn'] = AdvantageActorCritic
    config['discount'] = 0.99
    config['target_network_update_freq'] = 0
    config['step_limit'] = 10000
    config['n_workers'] = 16
    config['batch_size'] = 20
    config['test_interval'] = 50000
    config['test_repetitions'] = 1
    config['history_length'] = history_length
    config['logger'] = gym.logger
    agent = AsyncAgent(**config)
    agent.run()

if __name__ == '__main__':
    # gym.logger.setLevel(logging.DEBUG)
    gym.logger.setLevel(logging.INFO)

    # async_cart_pole()
    # dqn_cart_pole()
    # dqn_pixel_atari('BreakoutNoFrameskip-v3')
    # async_pixel_atari('BreakoutNoFrameskip-v3')
    # a3c_pixel_atari('BreakoutNoFrameskip-v3')
    # a3c_cart_pole()
    async_pixel_atari('PongNoFrameskip-v3')
    # a3c_pixel_atari('PongNoFrameskip-v3')
