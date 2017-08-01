import logging
from agent import *
from component import *
from utils import *

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
    config['step_limit'] = 200
    config['explore_steps'] = 1000
    config['logger'] = Logger('./log', gym.logger)
    config['history_length'] = 2
    config['test_interval'] = 100
    config['test_repetitions'] = 50
    # config['double_q'] = True
    config['double_q'] = False
    config['tag'] = ''
    agent = DQNAgent(**config)
    agent.run()

def async_cart_pole():
    config = Config()
    config.task_fn= lambda: CartPole()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: FCNet([4, 50, 200, 2])
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.5, final_step=5000, min_epsilon=0.1)
    config.worker = OneStepQLearning
    # config.worker = NStepQLearning
    # config.worker = OneStepSarsa
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.max_episode_length = 200
    config.num_workers = 16
    config.update_interval = 6
    config.test_interval = 1
    config.test_repetitions = 50
    config.logger = Logger('./log', gym.logger)
    agent = AsyncAgent(config)
    agent.run()

def a3c_cart_pole():
    config = Config()
    config.task_fn = lambda: CartPole()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: ActorCriticFCNet(4, 2)
    config.policy_fn = SamplePolicy
    config.worker = AdvantageActorCritic
    config.discount = 0.99
    config.max_episode_length = 200
    config.num_workers = 16
    config.update_interval = 6
    config.test_interval = 1
    config.test_repetitions = 30
    config.logger = Logger('./log', gym.logger)
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    agent = AsyncAgent(config)
    agent.run()

def a3c_pendulum():
    config = Config()
    config.task_fn = lambda: Pendulum()
    config.reward_shift_fn = lambda reward: reward / 10
    # config.task_fn = lambda: MountainCarContinuous()
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.0001)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: ContinuousActorCriticNet(
        task.state_dim, task.action_dim, 2, F.tanh)
    config.policy_fn = lambda: GaussianPolicy()
    config.worker = ContinuousAdvantageActorCritic
    config.discount = 0.99
    config.max_episode_length = 200
    config.num_workers = 8
    config.update_interval = 5
    config.test_interval = 1
    config.test_repetitions = 5
    config.entropy_weight = 0.0001
    config.gradient_clip = 40
    config.logger = Logger('./log', gym.logger)
    agent = AsyncAgent(config)
    agent.run()

def a3c_walker():
    config = Config()
    config.task_fn = lambda: BipedalWalker()
    shifter = Shifter()
    config.state_shift_fn = lambda state: shifter(state)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.0001)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: ContinuousActorCriticNet(
        task.state_dim, task.action_dim, 1, F.tanh)
    config.policy_fn = lambda: GaussianPolicy()
    config.worker = ContinuousAdvantageActorCritic
    config.discount = 0.99
    config.max_episode_length = 999
    config.num_workers = 8
    config.update_interval = 20
    config.test_interval = 1
    config.test_repetitions = 5
    config.entropy_weight = 0.01
    config.gradient_clip = 30
    config.logger = Logger('./log', gym.logger)
    agent = AsyncAgent(config)
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
    config['logger'] = Logger('./log', gym.logger)
    config['history_length'] = history_length
    config['test_interval'] = 10
    config['test_repetitions'] = 1
    # config['double_q'] = True
    config['double_q'] = False
    config['tag'] = ''
    agent = DQNAgent(**config)
    agent.run()

def async_pixel_atari(name):
    config = Config()
    config.history_length = 1
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, frame_size=42)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0001)
    config.network_fn = lambda: OpenAIConvNet(
        config.history_length, task.env.action_space.n)
    config.policy_fn = lambda: StochasticGreedyPolicy(
        epsilons=[0.7, 0.7, 0.7], final_step=2000000, min_epsilons=[0.1, 0.01, 0.5],
        probs=[0.4, 0.3, 0.3])
    # config.worker = OneStepSarsa
    # config.worker = NStepQLearning
    config.worker = OneStepQLearning
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 10000
    config.num_workers = 10
    config.update_interval = 20
    config.test_interval = 50000
    config.test_repetitions = 1
    config.logger = Logger('./log', gym.logger)
    agent = AsyncAgent(config)
    agent.run()

def a3c_pixel_atari(name):
    config = Config()
    config.history_length = 1
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, frame_size=42)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0001)
    config.network_fn = lambda: OpenAIActorCriticConvNet(
        config.history_length, task.env.action_space.n, LSTM=True)
    config.policy_fn = SamplePolicy
    config.worker = AdvantageActorCritic
    config.discount = 0.99
    config.max_episode_length = 10000
    config.num_workers = 10
    config.update_interval = 20
    config.test_interval = 50000
    config.test_repetitions = 1
    config.logger = Logger('./log', gym.logger)
    agent = AsyncAgent(config)
    agent.run()

def ddpg_pendulum():
    task_fn = lambda: Pendulum()
    task = task_fn()
    config = dict()
    config['task_fn'] = task_fn
    config['actor_network_fn'] = lambda: DDPGActorNet(task.state_dim, task.action_dim, F.tanh)
    config['critic_network_fn'] = lambda: DDPGCriticNet(task.state_dim, task.action_dim)
    config['actor_optimizer_fn'] = lambda params: torch.optim.Adam(params, lr=1e-4)
    config['critic_optimizer_fn'] =\
        lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
    config['replay_fn'] = lambda: HighDimActionReplay(memory_size=1000000, batch_size=64)
    config['discount'] = 0.99
    config['step_limit'] = 200
    config['tau'] = 0.001
    config['exploration_steps'] = 100
    config['random_process_fn'] = \
        lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.15, sigma=0.2)
    config['test_interval'] = 10
    config['test_repetitions'] = 10
    config['tag'] = ''
    config['logger'] = Logger('./log', gym.logger)
    agent = DDPGAgent(**config)
    agent.run()

def ddpg_bipedal_walker():
    task_fn = lambda: BipedalWalker()
    task = task_fn()
    config = dict()
    config['task_fn'] = task_fn
    config['actor_network_fn'] = lambda: DDPGActorNet(task.state_dim, task.action_dim, F.tanh, gpu=True)
    config['critic_network_fn'] = lambda: DDPGCriticNet(task.state_dim, task.action_dim, gpu=True)
    config['actor_optimizer_fn'] = lambda params: torch.optim.Adam(params, lr=1e-4)
    config['critic_optimizer_fn'] =\
        lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
    config['replay_fn'] = lambda: HighDimActionReplay(memory_size=1000000, batch_size=64)
    config['discount'] = 0.99
    config['step_limit'] = 1000
    config['tau'] = 0.001
    config['exploration_steps'] = 100
    config['noise_decay_steps'] = 10000
    config['random_process_fn'] = \
        lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.15, sigma=0.2)
    config['test_interval'] = 10
    config['test_repetitions'] = 10
    config['tag'] = ''
    config['logger'] = Logger('./log', gym.logger, True)
    agent = DDPGAgent(**config)
    agent.run()

if __name__ == '__main__':
    # gym.logger.setLevel(logging.DEBUG)
    gym.logger.setLevel(logging.INFO)

    # dqn_cart_pole()
    # async_cart_pole()
    # a3c_cart_pole()
    a3c_pendulum()
    # a3c_walker()

    # dqn_pixel_atari('PongNoFrameskip-v3')
    # async_pixel_atari('PongNoFrameskip-v3')
    # a3c_pixel_atari('PongNoFrameskip-v3')

    # dqn_pixel_atari('BreakoutNoFrameskip-v3')
    # async_pixel_atari('BreakoutNoFrameskip-v3')
    # a3c_pixel_atari('BreakoutNoFrameskip-v3')

    # ddpg_pendulum()
    # ddpg_bipedal_walker()