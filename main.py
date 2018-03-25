#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import logging
from agent import *
from component import *
from utils import *
import model.action_conditional_video_prediction as acvp

def dqn_cart_pole():
    config = Config()
    config.task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: FCNet([4, 50, 200, 2])
    # config.network_fn = lambda: DuelingFCNet([8, 50, 200, 2])
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    config.logger = Logger('./log', logger)
    config.test_interval = 100
    config.test_repetitions = 50
    config.double_q = True
    # config.double_q = False
    run_episodes(DQNAgent(config))

def async_cart_pole():
    config = Config()
    config.task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: FCNet([4, 50, 200, 2])
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.5, final_step=5000, min_epsilon=0.1)
    # config.worker = OneStepQLearning
    config.worker = NStepQLearning
    # config.worker = OneStepSarsa
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.num_workers = 16
    config.update_interval = 6
    config.test_interval = 1
    config.test_repetitions = 50
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def a3c_cart_pole():
    config = Config()
    name = 'CartPole-v0'
    # name = 'MountainCar-v0'
    config.task_fn = lambda: ClassicalControl(name, max_steps=200)
    # config.task_fn = lambda: LunarLander()
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: ActorCriticFCNet(task.state_dim, task.action_dim)
    config.policy_fn = SamplePolicy
    config.worker = AdvantageActorCritic
    config.discount = 0.99
    config.max_episode_length = 200
    config.num_workers = 7
    config.update_interval = 6
    config.test_interval = 1
    config.test_repetitions = 30
    config.logger = Logger('./log', logger)
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    agent = AsyncAgent(config)
    agent.run()

def a2c_cart_pole():
    config = Config()
    name = 'CartPole-v0'
    # name = 'MountainCar-v0'
    task_fn = lambda: ClassicalControl(name, max_steps=200)
    # task_fn = lambda: LunarLander()
    task = task_fn()
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: ActorCriticFCNet(task.state_dim, task.action_dim)
    config.policy_fn = SamplePolicy
    config.discount = 0.99
    config.test_interval = 200
    config.test_repetitions = 10
    config.logger = Logger('./log', logger)
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 20
    run_iterations(A2CAgent(config))

def dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, normalized_state=False,
                                        history_length=config.history_length)
    action_dim = config.task_fn().action_dim
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda: NatureConvNet(config.history_length, action_dim, gpu=0)
    # config.network_fn = lambda: DuelingNatureConvNet(config.history_length, action_dim)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=32, dtype=np.uint8)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 0
    config.exploration_steps= 50000
    config.logger = Logger('./log', logger)
    config.test_interval = 10
    config.test_repetitions = 1
    # config.double_q = True
    config.double_q = False
    run_episodes(DQNAgent(config))

def dqn_ram_atari(name):
    config = Config()
    config.history_length = 1
    config.task_fn = lambda: RamAtari(name, no_op=30, frame_skip=4)
    action_dim = config.task_fn().action_dim
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda: FCNet([128, 64, 64, action_dim], gpu=2)
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=1000000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32, dtype=np.uint8)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 0
    config.exploration_steps= 100
    config.logger = Logger('./log', logger)
    config.test_interval = 0
    config.test_repetitions = 10
    config.double_q = True
    # config.double_q = False
    run_episodes(DQNAgent(config))

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
    config.reward_shift_fn = lambda r: np.sign(r)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 10000
    config.num_workers = 6
    config.update_interval = 20
    config.test_interval = 50000
    config.test_repetitions = 1
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def a3c_pixel_atari(name):
    config = Config()
    config.history_length = 1
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, frame_size=42)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0001)
    config.network_fn = lambda: OpenAIActorCriticConvNet(
        config.history_length, task.env.action_space.n, LSTM=False)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.policy_fn = SamplePolicy
    config.worker = AdvantageActorCritic
    config.discount = 0.99
    config.num_workers = 6
    config.update_interval = 20
    config.test_interval = 50000
    config.test_repetitions = 1
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def a2c_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.num_workers = 5
    task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, frame_size=84,
                                 history_length=config.history_length)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    # config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0001)
    # config.network_fn = lambda: OpenAIActorCriticConvNet(
    config.network_fn = lambda: NatureActorCriticConvNet(
        config.history_length, task.task.env.action_space.n, gpu=3)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.policy_fn = SamplePolicy
    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.97
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.test_interval = 0
    config.iteration_log_interval = 100
    config.gradient_clip = 0.5
    config.logger = Logger('./log', logger, skip=True)
    run_iterations(A2CAgent(config))

def a3c_continuous():
    config = Config()
    config.task_fn = lambda: Pendulum()
    # config.task_fn = lambda: Box2DContinuous('BipedalWalker-v2')
    # config.task_fn = lambda: Box2DContinuous('BipedalWalkerHardcore-v2')
    # config.task_fn = lambda: Box2DContinuous('LunarLanderContinuous-v2')
    task = config.task_fn()
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, 0.0001)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: DisjointActorCriticNet(
        # lambda: GaussianActorNet(task.state_dim, task.action_dim, unit_std=False, action_gate=F.tanh, action_scale=2.0),
        lambda: GaussianActorNet(task.state_dim, task.action_dim, unit_std=True),
        lambda: GaussianCriticNet(task.state_dim))
    config.policy_fn = lambda: GaussianPolicy()
    config.worker = ContinuousAdvantageActorCritic
    config.discount = 0.99
    config.num_workers = 8
    config.update_interval = 20
    config.test_interval = 1
    config.test_repetitions = 1
    config.entropy_weight = 0
    config.gradient_clip = 40
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def p3o_continuous():
    config = Config()
    config.task_fn = lambda: Pendulum()
    # config.task_fn = lambda: Box2DContinuous('BipedalWalker-v2')
    # config.task_fn = lambda: Box2DContinuous('BipedalWalkerHardcore-v2')
    # config.task_fn = lambda: Box2DContinuous('LunarLanderContinuous-v2')
    # config.task_fn = lambda: Roboschool('RoboschoolInvertedPendulum-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolAnt-v1')
    task = config.task_fn()
    config.actor_network_fn = lambda: GaussianActorNet(task.state_dim, task.action_dim,
                                                       gpu=-1, unit_std=True)
    config.critic_network_fn = lambda: GaussianCriticNet(task.state_dim, gpu=-1)
    config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)

    config.policy_fn = lambda: GaussianPolicy()
    config.replay_fn = lambda: GeneralReplay(memory_size=2048, batch_size=2048)
    config.worker = ProximalPolicyOptimization
    config.discount = 0.99
    config.gae_tau = 0.97
    config.num_workers = 6
    config.test_interval = 1
    config.test_repetitions = 1
    config.entropy_weight = 0
    config.gradient_clip = 20
    config.rollout_length = 10000
    config.optimize_epochs = 1
    config.ppo_ratio_clip = 0.2
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def d3pg_continuous():
    config = Config()
    config.task_fn = lambda: Pendulum()
    # config.task_fn = lambda: Box2DContinuous('BipedalWalker-v2')
    # config.task_fn = lambda: Box2DContinuous('BipedalWalkerHardcore-v2')
    # config.task_fn = lambda: Box2DContinuous('LunarLanderContinuous-v2')
    # config.task_fn = lambda: Roboschool('RoboschoolInvertedPendulum-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolReacher-v1')
    task = config.task_fn()
    config.actor_network_fn = lambda: DeterministicActorNet(
        task.state_dim, task.action_dim, F.tanh, 2, non_linear=F.relu, batch_norm=False)
    config.critic_network_fn = lambda: DeterministicCriticNet(
        task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=False)
    config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.critic_optimizer_fn =\
        lambda params: torch.optim.Adam(params, lr=1e-4)
    config.replay_fn = lambda: SharedReplay(memory_size=1000000, batch_size=64,
                                            state_shape=(task.state_dim, ), action_shape=(task.action_dim, ))
    config.discount = 0.99
    config.random_process_fn = \
        lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.15, sigma=0.2,
                                         n_steps_annealing=100000)
    config.worker = DeterministicPolicyGradient
    config.num_workers = 6
    config.min_memory_size = 50
    config.target_network_mix = 0.001
    config.test_interval = 500
    config.test_repetitions = 1
    config.gradient_clip = 20
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def ddpg_continuous():
    config = Config()
    config.task_fn = lambda: Pendulum()
    # config.task_fn = lambda: Box2DContinuous('BipedalWalker-v2')
    # config.task_fn = lambda: Box2DContinuous('BipedalWalkerHardcore-v2')
    # config.task_fn = lambda: Box2DContinuous('LunarLanderContinuous-v2')
    # config.task_fn = lambda: Roboschool('RoboschoolInvertedPendulum-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolReacher-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolHopper-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolAnt-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolWalker2d-v1')
    task = config.task_fn()
    config.actor_network_fn = lambda: DeterministicActorNet(
        task.state_dim, task.action_dim, F.tanh, 1, non_linear=F.relu, batch_norm=False, gpu=-1)
    config.critic_network_fn = lambda: DeterministicCriticNet(
        task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=False, gpu=-1)
    config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.critic_optimizer_fn =\
        lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
    config.replay_fn = lambda: HighDimActionReplay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.random_process_fn = \
        lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.15, sigma=0.2,
                                         n_steps_annealing=100000)
    config.worker = DeterministicPolicyGradient
    config.min_memory_size = 50
    config.target_network_mix = 0.001
    config.test_interval = 0
    config.test_repetitions = 1
    config.gradient_clip = 40
    config.render_episode_freq = 0
    config.logger = Logger('./log', logger)
    run_episodes(DDPGAgent(config))

def categorical_dqn_cart_pole():
    config = Config()
    config.task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalFCNet(task.state_dim, task.action_dim, config.categorical_n_atoms)
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.logger = Logger('./log', logger, skip=True)
    # config.logger = Logger('./log', logger)
    config.test_interval = 100
    config.test_repetitions = 50
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 50
    run_episodes(CategoricalDQNAgent(config))

def categorical_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, normalized_state=False,
                                        history_length=config.history_length)
    action_dim = config.task_fn().action_dim
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalConvNet(config.history_length, action_dim, config.categorical_n_atoms, gpu=0)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=32, dtype=np.uint8)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = Logger('./log', logger)
    config.test_interval = 10
    config.test_repetitions = 1
    config.double_q = False
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    run_episodes(CategoricalDQNAgent(config))

def n_step_dqn_cart_pole():
    config = Config()
    task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    task = task_fn()
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: FCNet([task.state_dim, 50, 200, task.action_dim])
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 20
    config.logger = Logger('./log', logger)
    run_iterations(NStepDQNAgent(config))

def n_step_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, normalized_state=True,
                                 history_length=config.history_length)
    task = task_fn()
    config.num_workers = 8
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda: NatureConvNet(config.history_length, task.action_dim, gpu=0)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 20
    config.logger = Logger('./log', logger)
    run_iterations(NStepDQNAgent(config))

def quantile_regression_dqn_cart_pole():
    config = Config()
    config.task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: QuantileFCNet(task.state_dim, task.action_dim, config.num_quantiles)
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.logger = Logger('./log', logger, skip=True)
    # config.logger = Logger('./log', logger)
    config.test_interval = 100
    config.test_repetitions = 50
    config.num_quantiles = 20
    run_episodes(QuantileRegressionDQNAgent(config))

def quantile_regression_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, normalized_state=False,
                                        history_length=config.history_length)
    action_dim = config.task_fn().action_dim
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda: QuantileConvNet(config.history_length, action_dim, config.num_quantiles, gpu=0)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.01)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=32, dtype=np.uint8)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = Logger('./log', logger)
    config.test_interval = 10
    config.test_repetitions = 1
    config.double_q = False
    config.num_quantiles = 200
    run_episodes(QuantileRegressionDQNAgent(config))

if __name__ == '__main__':
    mkdir('data')
    mkdir('data/video')
    mkdir('log')
    os.system('export OMP_NUM_THREADS=1')
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # dqn_cart_pole()
    # categorical_dqn_cart_pole()
    # quantile_regression_dqn_cart_pole()
    # async_cart_pole()
    # a3c_cart_pole()
    a2c_cart_pole()
    # a3c_continuous()
    # p3o_continuous()
    # d3pg_continuous()
    # ddpg_continuous()
    # n_step_dqn_cart_pole()

    # dqn_pixel_atari('PongNoFrameskip-v4')
    # categorical_dqn_pixel_atari('PongNoFrameskip-v4')
    # quantile_regression_dqn_pixel_atari('PongNoFrameskip-v4')
    # n_step_dqn_pixel_atari('PongNoFrameskip-v4')
    # async_pixel_atari('PongNoFrameskip-v4')
    # a3c_pixel_atari('PongNoFrameskip-v4')
    # a2c_pixel_atari('PongNoFrameskip-v4')

    # dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # async_pixel_atari('BreakoutNoFrameskip-v4')
    # a3c_pixel_atari('BreakoutNoFrameskip-v4')

    # dqn_ram_atari('Pong-ramNoFrameskip-v4')

    # acvp.train('PongNoFrameskip-v4')

