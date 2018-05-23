from deep_rl import *

def quantile_regression_dqn_cart_pole():
    config = Config()
    config.task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    config.evaluation_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, FCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.logger = get_logger(skip=True)
    config.num_quantiles = 20
    run_episodes(QuantileRegressionDQNAgent(config))

def quantile_regression_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(quantile_regression_dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, NatureConvBody(), gpu=1)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.01)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(file_name=quantile_regression_dqn_pixel_atari.__name__)
    config.double_q = False
    config.num_quantiles = 200
    run_episodes(QuantileRegressionDQNAgent(config))

def qr_dqn_cart_pole():
    config = Config()
    task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    # config.evaluation_env = task_fn(None)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, FCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.logger = get_logger()
    config.num_quantiles = 20
    run_iterations(NStepQRDQNAgent(config))

def option_qr_dqn_cart_pole():
    config = Config()
    task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    # config.evaluation_env = task_fn(None)
    config.num_workers = 5
    config.num_options = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: \
        OptionQuantileNet(action_dim, config.num_quantiles, config.num_options, FCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.discount = 0.99
    config.entropy_weight = 0.01
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.logger = get_logger()
    config.num_quantiles = 20
    run_iterations(OptionNStepQRDQNAgent(config))

def n_step_dqn_pixel_atari(name, **kwargs):
    config = Config()
    kwargs.setdefault('gpu', 0)
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(n_step_dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody(), gpu=kwargs['gpu'])
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=100000, min_epsilon=0.05)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.logger = get_logger()
    run_iterations(NStepDQNAgent(config))

def qr_dqn_pixel_atari(name, **kwargs):
    config = Config()
    kwargs.setdefault('tag', qr_dqn_pixel_atari.__name__)
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('random_skip', 0)
    kwargs.setdefault('frame_stack', 4)
    config.history_length = kwargs['frame_stack']
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                         log_dir=log_dir, random_skip=kwargs['random_skip'])
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=kwargs['log_dir'], single_process=True)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, NatureConvBody(in_channels=config.history_length), gpu=kwargs['gpu'])
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=100000, min_epsilon=0.05)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(1e8)
    config.logger = get_logger(file_name=qr_dqn_pixel_atari.__name__)
    config.num_quantiles = 200
    config.merge(kwargs)
    run_iterations(NStepQRDQNAgent(config))

def option_qr_dqn_pixel_atari(name, **kwargs):
    config = Config()
    kwargs.setdefault('tag', option_qr_dqn_pixel_atari.__name__)
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('mean_option', 1)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('random_skip', 0)
    kwargs.setdefault('frame_stack', 4)
    config.history_length = kwargs['frame_stack']
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                         log_dir=log_dir, random_skip=kwargs['random_skip'])
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=kwargs['log_dir'], single_process=True)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: \
        OptionQuantileNet(action_dim, config.num_quantiles, kwargs['num_options'] + kwargs['mean_option'],
                          NatureConvBody(in_channels=config.history_length),
                          gpu=kwargs['gpu'])
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=100000, min_epsilon=0.05)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.entropy_weight = 0.01
    config.max_steps = int(1e8)
    config.logger = get_logger(file_name=option_qr_dqn_pixel_atari.__name__)
    config.num_quantiles = 200
    config.merge(kwargs)
    run_iterations(OptionNStepQRDQNAgent(config))

def single_run(run, game, fn, tag, **kwargs):
    log_dir = './log/dist_rl-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
    fn(game, log_dir=log_dir, tag=tag, **kwargs)

def multi_runs(game, fn, tag, **kwargs):
    runs = np.arange(0, 2)
    kwargs.setdefault('parallel', False)
    if not kwargs['parallel']:
        for run in runs:
            single_run(run, game, fn, tag, **kwargs)
        return
    ps = [mp.Process(target=single_run, args=(run, game, fn, tag), kwargs=kwargs) for run in runs]
    for p in ps:
        p.start()
        time.sleep(1)
    for p in ps: p.join()

if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    # game = 'BreakoutNoFrameskip-v4'
    # game = 'FreewayNoFrameskip-v4'
    # game = 'SeaquestNoFrameskip-v4'
    # game = 'MsPacmanNoFrameskip-v4'
    # game = 'FrostbiteNoFrameskip-v4'
    # game = 'EnduroNoFrameskip-v4'
    # game = 'JourneyEscapeNoFrameskip-v4'
    # game = 'SolarisNoFrameskip-v4'
    # game = 'TennisNoFrameskip-v4'
    # game = 'PongNoFrameskip-v4'
    # game = 'BoxingNoFrameskip-v4'
    game = 'IceHockeyNoFrameskip-v4'

    # option_qr_dqn_cart_pole()
    # qr_dqn_cart_pole()

    # n_step_dqn_pixel_atari(game)
    # qr_dqn_pixel_atari(game)
    # quantile_regression_dqn_pixel_atari(game)

    # qr_dqn_pixel_atari(game, gpu=0, tag='qr_dqn')
    # option_qr_dqn_pixel_atari(game, num_options=9, gpu=1)

    # option_qr_dqn_pixel_atari(game, num_options=20, gpu=0, tag='option_qr_20_options')
    # option_qr_dqn_pixel_atari(game, num_options=5, gpu=1, tag='option_qr_5_options')

    # multi_runs(game, option_qr_dqn_pixel_atari, num_options=9, gpu=0, tag='mean_and_9_options', parallel=True)
    multi_runs(game, qr_dqn_pixel_atari, gpu=0, tag='original_qr_dqn', parallel=True)

    # multi_runs(game, option_qr_dqn_pixel_atari, num_options=9, gpu=1, tag='mean_and_9_options_random_skip', parallel=True)
    # multi_runs(game, qr_dqn_pixel_atari, gpu=1, tag='qr_dqn_random_skip', parallel=True)
