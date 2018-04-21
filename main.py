#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import logging
from agent import *
from component import *
from utils import *
from model import *

## cart pole

def dqn_cart_pole():
    game = 'CartPole-v0'
    config = Config()
    config.task_fn = lambda: ClassicalControl(game, max_steps=200)
    config.evaluation_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, TwoLayerFCBody(state_dim))
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, TwoLayerFCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    config.logger = Logger('./log', logger)
    config.double_q = True
    # config.double_q = False
    run_episodes(DQNAgent(config))

def a2c_cart_pole():
    config = Config()
    name = 'CartPole-v0'
    # name = 'MountainCar-v0'
    task_fn = lambda log_dir: ClassicalControl(name, max_steps=200, log_dir=log_dir)
    config.evaluation_env = task_fn(None)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(a2c_cart_pole.__name__))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: ActorCriticNet(action_dim, TwoLayerFCBody(state_dim))
    config.policy_fn = SamplePolicy
    config.discount = 0.99
    config.logger = Logger('./log', logger)
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    run_iterations(A2CAgent(config))

def categorical_dqn_cart_pole():
    game = 'CartPole-v0'
    config = Config()
    config.task_fn = lambda: ClassicalControl(game, max_steps=200)
    config.evaluation_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: \
        CategoricalNet(action_dim, config.categorical_n_atoms, TwoLayerFCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.logger = Logger('./log', logger, skip=True)
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 50
    run_episodes(CategoricalDQNAgent(config))

def quantile_regression_dqn_cart_pole():
    config = Config()
    config.task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    config.evaluation_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, TwoLayerFCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.logger = Logger('./log', logger, skip=True)
    config.num_quantiles = 20
    run_episodes(QuantileRegressionDQNAgent(config))

def n_step_dqn_cart_pole():
    config = Config()
    task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    config.evaluation_env = task_fn(None)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, TwoLayerFCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.logger = Logger('./log', logger)
    run_iterations(NStepDQNAgent(config))

def ppo_cart_pole():
    config = Config()
    task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    network_fn = lambda state_dim, action_dim: ActorCriticNet(action_dim, TwoLayerFCBody(state_dim))
    config.network_fn = lambda state_dim, action_dim: \
        CategoricalActorCriticWrapper(state_dim, action_dim, network_fn, optimizer_fn)
    config.discount = 0.99
    config.logger = Logger('./log', logger)
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 10
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    run_iterations(PPOAgent(config))

## Atari games

def dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody(), gpu=0)
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody(), gpu=0)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = Logger('./log', logger)
    # config.double_q = True
    config.double_q = False
    run_episodes(DQNAgent(config))

def a2c_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.num_workers = 16
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=get_default_log_dir(a2c_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda state_dim, action_dim: \
        ActorCriticNet(action_dim, NatureConvBody(), gpu=1)
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.97
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.logger = Logger('./log', logger, skip=True)
    run_iterations(A2CAgent(config))

def categorical_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(categorical_dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda state_dim, action_dim: \
        CategoricalNet(action_dim, config.categorical_n_atoms, NatureConvBody(), gpu=1)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = Logger('./log', logger)
    config.double_q = False
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    run_episodes(CategoricalDQNAgent(config))

def quantile_regression_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(quantile_regression_dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, NatureConvBody(), gpu=2)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.01)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = Logger('./log', logger)
    config.double_q = False
    config.num_quantiles = 200
    run_episodes(QuantileRegressionDQNAgent(config))

def n_step_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(n_step_dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody(), gpu=3)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.05)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.logger = Logger('./log', logger)
    run_iterations(NStepDQNAgent(config))

def ppo_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(ppo_pixel_atari.__name__))
    optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025)
    network_fn = lambda state_dim, action_dim: ActorCriticNet(action_dim, NatureConvBody(), gpu=2)
    config.network_fn = lambda state_dim, action_dim: \
        CategoricalActorCriticWrapper(state_dim, action_dim, network_fn, optimizer_fn)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.logger = Logger('./log', logger)
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    run_iterations(PPOAgent(config))

def dqn_ram_atari(name):
    config = Config()
    config.task_fn = lambda: RamAtari(name, no_op=30, frame_skip=4,
                                      log_dir=get_default_log_dir(dqn_ram_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, TwoLayerFCBody(state_dim), gpu=2)
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=1000000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
    config.state_normalizer = RescaleNormalizer(1.0 / 128)
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 0
    config.exploration_steps= 100
    config.logger = Logger('./log', logger)
    config.double_q = True
    # config.double_q = False
    run_episodes(DQNAgent(config))

## continuous control

def ppo_continuous():
    config = Config()
    config.num_workers = 1
    # task_fn = lambda log_dir: Pendulum(log_dir=log_dir)
    # task_fn = lambda log_dir: Roboschool('RoboschoolInvertedPendulum-v1', log_dir=log_dir)
    task_fn = lambda log_dir: Roboschool('RoboschoolAnt-v1', log_dir=log_dir)
    # task_fn = lambda log_dir: Roboschool('RoboschoolReacher-v1', log_dir=log_dir)
    # task_fn = lambda log_dir: Roboschool('RoboschoolHopper-v1', log_dir=log_dir)
    # task_fn = lambda log_dir: DMControl('cartpole', 'balance', log_dir=log_dir)
    # task_fn = lambda log_dir: DMControl('hopper', 'hop', log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=get_default_log_dir(ppo_continuous.__name__))
    actor_network_fn = lambda state_dim, action_dim: GaussianActorNet(state_dim, action_dim)
    critic_network_fn = lambda state_dim: GaussianCriticNet(state_dim)
    actor_optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    critic_optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: \
        GaussianActorCriticWrapper(state_dim, action_dim, actor_network_fn,
                                     critic_network_fn, actor_optimizer_fn,
                                     critic_optimizer_fn)
    # config.state_normalizer = RunningStatsNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.num_mini_batches = 32
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    config.logger = Logger('./log', logger)
    run_iterations(PPOAgent(config))

def ddpg_continuous():
    config = Config()
    log_dir = get_default_log_dir(ddpg_continuous.__name__)
    # config.task_fn = lambda: Pendulum(log_dir=log_dir)
    # config.task_fn = lambda: Roboschool('RoboschoolInvertedPendulum-v1', log_dir=log_dir)
    # config.task_fn = lambda: Roboschool('RoboschoolReacher-v1', log_dir=log_dir)
    config.task_fn = lambda: Roboschool('RoboschoolHopper-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolAnt-v1', log_dir=log_dir)
    # config.task_fn = lambda: Roboschool('RoboschoolWalker2d-v1', log_dir=log_dir)
    # config.task_fn = lambda: DMControl('cartpole', 'balance', log_dir=log_dir)
    # config.task_fn = lambda: DMControl('finger', 'spin', log_dir=log_dir)
    config.evaluation_env = Roboschool('RoboschoolHopper-v1', log_dir=log_dir)
    config.actor_network_fn = lambda state_dim, action_dim: DeterministicActorNet(state_dim, action_dim)
    config.critic_network_fn = lambda state_dim, action_dim: DeterministicCriticNet(state_dim, action_dim)
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.random_process_fn = \
        lambda action_dim: OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, sigma=0.3,
                                         n_steps_annealing=1000000)
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.gradient_clip = 1.0
    config.logger = Logger('./log', logger)
    run_episodes(DDPGAgent(config))

def plot():
    import matplotlib.pyplot as plt
    plotter = Plotter()
    # name = 'log/ppo_continuous-180408-002056'
    # plotter.plot_results([name])
    # plt.show()
    names = [
            # 'a2c_pixel_atari-180407-92711',
            # 'categorical_dqn_pixel_atari-180407-094006',
            # 'dqn_pixel_atari-180407-01414',
            # 'quantile_regression_dqn_pixel_atari-180407-01604',
            #  'n_step_dqn_pixel_atari-180408-001104',
            #  'ppo_continuous-180408-002056',
            #  'ddpg_continuous-180407-234141'
            'ppo_pixel_atari-180410-235529',
             ]
    for name in names:
        plotter.plot_results(['to_plot/%s' % (name)], title='BreakoutNoFrameskip-v4')
        plt.savefig('images/%s.png' % (name))
        plt.close()

def action_conditional_video_prediction():
    game = 'PongNoFrameskip-v4'
    prefix = '.'

    # Train an agent to generate the dataset
    # a2c_pixel_atari(game)

    # Generate a dataset with the trained model
    # a2c_model_file = './data/A2CAgent-vanilla-model-%s.bin' % (game)
    # generate_dataset(game, a2c_model_file, prefix)

    # Train the action conditional video prediction model
    acvp_train(game, prefix)


if __name__ == '__main__':
    mkdir('data')
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    os.system('export OMP_NUM_THREADS=1')
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # dqn_cart_pole()
    # a2c_cart_pole()
    # categorical_dqn_cart_pole()
    # quantile_regression_dqn_cart_pole()
    # n_step_dqn_cart_pole()
    # ppo_cart_pole()

    # dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # a2c_pixel_atari('BreakoutNoFrameskip-v4')
    # categorical_dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # quantile_regression_dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # n_step_dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # ppo_pixel_atari('BreakoutNoFrameskip-v4')
    # dqn_ram_atari('Breakout-ramNoFrameskip-v4')

    # ddpg_continuous()
    # ppo_continuous()

    # action_conditional_video_prediction()

    # plot()

