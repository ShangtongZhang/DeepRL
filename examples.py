#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *

# DQN
def dqn_cart_pole():
    game = 'CartPole-v0'
    config = Config()
    config.task_fn = lambda: ClassicalControl(game, max_steps=200)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))

    # config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e4), batch_size=10)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    # config.async_actor = False
    config.logger = get_logger()
    run_steps(DQNAgent(config))

def dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.eval_env = PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                 episode_life=False)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    # config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    # config.double_q = True
    config.double_q = False
    config.max_steps = int(2e7)
    config.logger = get_logger(file_name=dqn_pixel_atari.__name__)
    run_steps(DQNAgent(config))

def dqn_ram_atari(name):
    config = Config()
    config.task_fn = lambda: RamAtari(name, no_op=30, frame_skip=4,
                                      log_dir=get_default_log_dir(dqn_ram_atari.__name__))
    config.eval_env = RamAtari(name, no_op=30, frame_skip=4, episode_life=False)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    config.random_action_prob = LinearSchedule(0.1)
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = RescaleNormalizer(1.0 / 128)
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 0
    config.exploration_steps= 100
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = True
    config.logger = get_logger()
    run_steps(DQNAgent(config))

# QR DQN
def quantile_regression_dqn_cart_pole():
    config = Config()
    config.task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: QuantileNet(config.action_dim, config.num_quantiles, FCBody(config.state_dim))

    # config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e4), batch_size=10)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.num_quantiles = 20
    config.gradient_clip = 5
    config.sgd_update_frequency = 4
    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    config.logger = get_logger()
    run_steps(QuantileRegressionDQNAgent(config))

def quantile_regression_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(quantile_regression_dqn_pixel_atari.__name__))
    config.eval_env = PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                 episode_life=False)

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda: QuantileNet(config.action_dim, config.num_quantiles, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.num_quantiles = 200
    config.max_steps = int(2e7)
    config.logger = get_logger(file_name=quantile_regression_dqn_pixel_atari.__name__)
    run_steps(QuantileRegressionDQNAgent(config))

# C51
def categorical_dqn_cart_pole():
    game = 'CartPole-v0'
    config = Config()
    config.task_fn = lambda: ClassicalControl(game, max_steps=200)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, FCBody(config.state_dim))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)

    # config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=10000, batch_size=10)

    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 50
    config.gradient_clip = 5
    config.sgd_update_frequency = 4

    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    config.logger = get_logger()
    run_steps(CategoricalDQNAgent(config))

def categorical_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(categorical_dqn_pixel_atari.__name__))
    config.eval_env = PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                 episode_life=False)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    config.sgd_update_frequency = 4
    config.gradient_clip = 0.5
    config.max_steps = int(2e7)
    config.logger = get_logger(file_name=categorical_dqn_pixel_atari.__name__)
    run_steps(CategoricalDQNAgent(config))

# A2C
def a2c_cart_pole():
    config = Config()
    name = 'CartPole-v0'
    # name = 'MountainCar-v0'
    task_fn = lambda log_dir: ClassicalControl(name, max_steps=200, log_dir=log_dir)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(a2c_cart_pole.__name__))
    config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim))
    config.discount = 0.99
    config.logger = get_logger()
    config.use_gae = False
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    run_steps(A2CAgent(config))

def a2c_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.num_workers = 16
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(
        task_fn, config.num_workers, log_dir=get_default_log_dir(a2c_pixel_atari.__name__),
        single_process=True)
    config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.logger = get_logger(file_name=a2c_pixel_atari.__name__)
    run_steps(A2CAgent(config))

def a2c_continuous():
    config = Config()
    config.history_length = 4
    config.num_workers = 16
    task_fn = lambda log_dir: Roboschool('RoboschoolHopper-v1', log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(
        task_fn, config.num_workers, log_dir=get_default_log_dir(a2c_continuous.__name__),
        single_process=True)
    config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.logger = get_logger(file_name=a2c_continuous.__name__)
    run_steps(A2CAgent(config))

# N-Step DQN
def n_step_dqn_cart_pole():
    config = Config()
    task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    config.eval_env = task_fn(None)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.gradient_clip = 5
    config.logger = get_logger()
    run_steps(NStepDQNAgent(config))

def n_step_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(n_step_dqn_pixel_atari.__name__),
                                              single_process=True)
    config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.05, 1e6)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.logger = get_logger(file_name=n_step_dqn_pixel_atari.__name__)
    run_steps(NStepDQNAgent(config))

# Option-Critic
def option_critic_cart_pole():
    config = Config()
    game = 'CartPole-v0'
    task_fn = lambda log_dir: ClassicalControl(game, max_steps=200, log_dir=log_dir)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: OptionCriticNet(FCBody(config.state_dim), config.action_dim, num_options=2)
    config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.logger = get_logger()
    run_steps(OptionCriticAgent(config))

def option_ciritc_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(option_ciritc_pixel_atari.__name__),
                                              single_process=True)
    config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticNet(NatureConvBody(), config.action_dim, num_options=4)
    config.random_option_prob = LinearSchedule(0.1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    config.logger = get_logger(file_name=option_ciritc_pixel_atari.__name__)
    run_steps(OptionCriticAgent(config))

# PPO
def ppo_cart_pole():
    config = Config()
    task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, FCBody(config.state_dim))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.rollout_length = 128
    config.optimization_epochs = 10
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.2
    config.log_interval = 128 * 5 * 10
    config.logger = get_logger()
    run_steps(PPOAgent(config))

def ppo_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(ppo_pixel_atari.__name__),
                                              single_process=True)
    config.eval_env = PixelAtari(name, frame_skip=4, history_length=config.history_length, episode_life=False)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 16
    config.max_steps = int(2e7)
    config.logger = get_logger(file_name=ppo_pixel_atari.__name__)
    run_steps(PPOAgent(config))

def ppo_continuous():
    config = Config()
    config.num_workers = 1
    # task_fn = lambda log_dir: Pendulum(log_dir=log_dir)
    # task_fn = lambda log_dir: Bullet('AntBulletEnv-v0', log_dir=log_dir)
    task_fn = lambda log_dir: Roboschool('RoboschoolHopper-v1', log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=get_default_log_dir(ppo_continuous.__name__))
    config.eval_env = task_fn(None)

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.num_mini_batches = 32
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 2e7
    config.logger = get_logger()
    run_steps(PPOAgent(config))

# DDPG
def ddpg_low_dim_state():
    config = Config()
    log_dir = get_default_log_dir(ddpg_low_dim_state.__name__)
    # config.task_fn = lambda **kwargs: Pendulum(log_dir=log_dir)
    # config.task_fn = lambda **kwargs: Bullet('AntBulletEnv-v0', **kwargs)
    config.task_fn = lambda **kwargs: Roboschool('RoboschoolHopper-v1', **kwargs)
    config.eval_env = config.task_fn(log_dir=log_dir)
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.tanh),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=F.tanh),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim, ), std=LinearSchedule(0.2))
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    run_steps(DDPGAgent(config))

def ddpg_pixel():
    config = Config()
    log_dir = get_default_log_dir(ddpg_pixel.__name__)
    config.task_fn = lambda **kwargs: CarRacing(**kwargs)
    config.eval_env = config.task_fn(log_dir=None)

    phi_body = NatureConvBody()
    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim, phi_body=phi_body,
        actor_body=FCBody(phi_body.feature_dim, (50, ), gate=F.relu),
        critic_body=OneLayerFCBodyWithAction(
            phi_body.feature_dim, config.action_dim, 50, gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.max_steps = 1e7
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim, ), std=LinearSchedule(0.2))
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.log_interval = 100
    config.logger = get_logger(file_name=ddpg_pixel.__name__)
    run_steps(DDPGAgent(config))

def plot():
    import matplotlib.pyplot as plt
    plotter = Plotter()
    dirs = [
        'a2c_pixel_atari-180629-211609',
        'dqn_pixel_atari-180628-052904',
        'n_step_dqn_pixel_atari-180628-053553',
        'option_ciritc_pixel_atari-180628-053626',
        'ppo_pixel_atari-180628-053657',
        'quantile_regression_dqn_pixel_atari-180628-053044',
        'categorical_dqn_pixel_atari-180629-040601'
    ]
    names = [
        'A2C',
        'DQN',
        'NStepDQN',
        'OptionCritic',
        'PPO',
        'QRDQN',
        'C51'
    ]

    plt.figure(0)
    for i, dir in enumerate(dirs):
        data = plotter.load_results(['./images_data/%s' % (dir)], episode_window=100)
        x, y = data[0]
        plt.plot(x, y, label=names[i])
    plt.xlabel('steps')
    plt.ylabel('episode return')
    plt.legend()

    plt.figure(1)
    plt.subplot(1, 2, 1)
    x, y = plotter.load_evaluation_episodes_results(['./images_data/ddpg_low_dim_state-180628-110025'],
                                                    int(1e4), 20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.plot(x, y.mean(0), label='DDPG')
    plt.xlabel('steps')
    plt.ylabel('episode return')
    plt.legend()

    plt.subplot(1, 2, 2)
    data = plotter.load_results(['./images_data/ppo_continuous-180630-035604'], episode_window=100)
    x, y = data[0]
    plt.plot(x, y, label='PPO')
    plt.xlabel('steps')
    plt.legend()

    plt.show()

def action_conditional_video_prediction():
    game = 'PongNoFrameskip-v4'
    prefix = '.'

    # Train an agent for generating the dataset
    # Make sure to set config.save_interval to non-zero value
    # a2c_pixel_atari(game)

    # Generate a dataset with the trained model
    # a2c_model_file = /path/to/the/saved/model
    # generate_dataset(game, a2c_model_file, prefix)

    # Train the action conditional video prediction model
    # acvp_train(game, prefix)

if __name__ == '__main__':
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(-1)
    # select_device(0)

    # dqn_cart_pole()
    # quantile_regression_dqn_cart_pole()
    # categorical_dqn_cart_pole()
    # a2c_cart_pole()
    # a2c_continuous()
    # n_step_dqn_cart_pole()
    # option_critic_cart_pole()
    # ppo_cart_pole()
    # ppo_continuous()
    # ddpg_low_dim_state()

    game = 'Breakout'
    # dqn_pixel_atari(game)
    # quantile_regression_dqn_pixel_atari(game)
    # categorical_dqn_pixel_atari(game)
    # a2c_pixel_atari(game)
    # n_step_dqn_pixel_atari(game)
    # option_ciritc_pixel_atari(game)
    # ppo_pixel_atari(game)
    # dqn_ram_atari(game)
    # ddpg_pixel()

    # action_conditional_video_prediction()

    # plot()

