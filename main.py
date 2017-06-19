from async_agent import *
from dqn_agent import *
import logging
import traceback

def dqn_cart_pole():
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.RMSprop(params, 0.001)
    config['network_fn'] = lambda optimizer_fn: FCNet([8, 50, 200, 2], optimizer_fn)
    # config['network_fn'] = lambda optimizer_fn: DuelingFCNet([8, 50, 200, 2], optimizer_fn)
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
    # config['double_q'] = True
    config['double_q'] = False
    agent = DQNAgent(**config)
    agent.run()

def async_cart_pole():
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, 0.001)
    config['network_fn'] = lambda: FCNet([4, 50, 200, 2])
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=0.5, final_step=5000, min_epsilon=0.1)
    config['worker_fn'] = OneStepQLearning
    # config['worker_fn'] = NStepQLearning
    # config['worker_fn'] = OneStepSarsa
    config['discount'] = 0.99
    config['target_network_update_freq'] = 200
    config['step_limit'] = 0
    config['n_workers'] = 16
    config['update_interval'] = 6
    config['test_interval'] = 4000
    config['test_repetitions'] = 50
    config['history_length'] = 1
    config['logger'] = gym.logger
    agent = AsyncAgent(**config)
    agent.run()

def a3c_cart_pole():
    update_interval = 6
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, 0.001)
    config['network_fn'] = lambda: ActorCriticFCNet([4, 200, 2])
    config['policy_fn'] = SamplePolicy
    config['worker_fn'] = AdvantageActorCritic
    config['discount'] = 0.99
    config['target_network_update_freq'] = 200
    config['step_limit'] = 0
    config['n_workers'] = 16
    config['update_interval'] = update_interval
    config['history_length'] = 1
    config['test_interval'] = 4000
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
    config['network_fn'] = lambda optimizer_fn: NatureConvNet(history_length, n_actions, optimizer_fn)
    # config['network_fn'] = lambda optimizer_fn: DuelingNatureConvNet(history_length, n_actions, optimizer_fn)
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config['replay_fn'] = lambda: Replay(memory_size=1000000, batch_size=32, dtype=np.uint8)
    config['discount'] = 0.99
    config['target_network_update_freq'] = 10000
    config['step_limit'] = 0
    config['explore_steps'] = 50000
    config['logger'] = gym.logger
    config['history_length'] = history_length
    config['test_interval'] = 10
    config['test_repetitions'] = 1
    # config['double_q'] = True
    config['double_q'] = False
    agent = DQNAgent(**config)
    agent.tag = 'dueling_'
    agent.run()

def async_pixel_atari(name):
    config = dict()
    history_length = 1
    n_actions = 6
    config['task_fn'] = lambda: PixelAtari(name, no_op=30, frame_skip=4, frame_size=42)
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, lr=0.0001)
    config['network_fn'] = lambda: OpenAIConvNet(history_length,
                                                 n_actions)
    config['policy_fn'] = lambda: StochasticGreedyPolicy(epsilons=[0.7, 0.7, 0.7],
                                                          final_step=2000000,
                                                          min_epsilons=[0.1, 0.01, 0.5],
                                                          probs=[0.4, 0.3, 0.3])
    # config['worker_fn'] = OneStepQLearning
    # config['worker_fn'] = NStepQLearning
    config['worker_fn'] = OneStepSarsa
    config['discount'] = 0.99
    config['target_network_update_freq'] = 10000
    config['step_limit'] = 10000
    config['n_workers'] = 16
    config['update_interval'] = 20
    config['test_interval'] = 50000
    config['test_repetitions'] = 1
    config['history_length'] = history_length
    config['logger'] = gym.logger
    agent = AsyncAgent(**config)
    agent.tag = 'Centered-target-network-'
    agent.run()

def a3c_pixel_atari(name):
    config = dict()
    history_length = 1
    n_actions = 6
    config['task_fn'] = lambda: PixelAtari(name, no_op=30, frame_skip=4, frame_size=42)
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, lr=0.0001)
    config['network_fn'] = lambda: OpenAIActorCriticConvNet(history_length,
                                                            n_actions,
                                                            LSTM=False)
    config['policy_fn'] = SamplePolicy
    config['worker_fn'] = AdvantageActorCritic
    config['discount'] = 0.99
    config['target_network_update_freq'] = 0
    config['step_limit'] = 10000
    config['n_workers'] = 16
    config['update_interval'] = 20
    config['test_interval'] = 50000
    config['test_repetitions'] = 1
    config['history_length'] = history_length
    config['logger'] = gym.logger
    agent = AsyncAgent(**config)
    agent.tag = ''
    agent.run()

if __name__ == '__main__':
    gym.logger.setLevel(logging.DEBUG)
    # gym.logger.setLevel(logging.INFO)

    # dqn_cart_pole()
    # async_cart_pole()
    # a3c_cart_pole()

    # dqn_pixel_atari('PongNoFrameskip-v3')
    async_pixel_atari('PongNoFrameskip-v3')
    # a3c_pixel_atari('PongNoFrameskip-v3')

    # dqn_pixel_atari('BreakoutNoFrameskip-v3')
    # async_pixel_atari('BreakoutNoFrameskip-v3')
    # a3c_pixel_atari('BreakoutNoFrameskip-v3')
