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

def quantile_regression_dqn_pixel_atari(game, **kwargs):
    config = Config()
    kwargs.setdefault('tag', quantile_regression_dqn_pixel_atari.__name__)
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(game, frame_skip=4, history_length=config.history_length,
                                        log_dir=kwargs['log_dir'])
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, NatureConvBody(), gpu=config.gpu)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.01)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(file_name=kwargs['tag'])
    config.double_q = False
    config.num_quantiles = 200
    config.merge(kwargs)
    run_episodes(QuantileRegressionDQNAgent(config))

def option_quantile_regression_dqn_pixel_atari(game, **kwargs):
    config = Config()
    kwargs.setdefault('tag', option_quantile_regression_dqn_pixel_atari.__name__)
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('mean_option', 1)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(game, frame_skip=4, history_length=config.history_length,
                                        log_dir=kwargs['log_dir'])
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda state_dim, action_dim: \
        OptionQuantileNet(action_dim, config.num_quantiles, config.num_options + config.mean_option,
                          NatureConvBody(), gpu=config.gpu)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.01)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(file_name=kwargs['tag'])
    config.double_q = False
    config.num_quantiles = 200
    config.entropy_weight = 0.01
    config.merge(kwargs)
    run_episodes(OptionQuantileRegressionDQNAgent(config))

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

def qr_dqn_cliff(**kwargs):
    kwargs.setdefault('tag', option_qr_dqn_cliff.__name__)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    config = Config()
    config.merge(kwargs)
    task_fn = lambda log_dir: CliffWalkingTask(random_action_prob=0.1, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=kwargs['log_dir'],
                                              single_process=True)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.01)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, FCBody(state_dim, hidden_units=(128, ), gate=F.relu))
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=config.max_steps, min_epsilon=0.1)
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.logger = get_logger()
    config.num_quantiles = 20
    config.max_steps = int(3e5)
    run_iterations(NStepQRDQNAgent(config))

def option_qr_dqn_cliff(**kwargs):
    kwargs.setdefault('tag', option_qr_dqn_cliff.__name__)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('random_option', False)
    kwargs.setdefault('mean_option', False)
    kwargs.setdefault('num_options', 5)
    config = Config()
    config.merge(kwargs)
    task_fn = lambda log_dir: CliffWalkingTask(random_action_prob=0.1, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=kwargs['log_dir'],
                                              single_process=True)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.01)
    config.network_fn = lambda state_dim, action_dim: \
        OptionQuantileNet(action_dim, config.num_quantiles, config.num_options + config.mean_option, FCBody(state_dim, hidden_units=(128, ), gate=F.relu))
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.1, final_step=config.max_steps, min_epsilon=0.1)
    config.entropy_weight = 0.01
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.logger = get_logger()
    config.num_quantiles = 20
    config.max_steps = int(3e5)
    run_iterations(OptionNStepQRDQNAgent(config))

def n_step_dqn_pixel_atari(game, **kwargs):
    config = Config()
    kwargs.setdefault('gpu', 0)
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(game, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
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

def qr_dqn_pixel_atari(game, **kwargs):
    config = Config()
    kwargs.setdefault('tag', qr_dqn_pixel_atari.__name__)
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('random_skip', 0)
    kwargs.setdefault('frame_stack', 4)
    kwargs.setdefault('max_steps', 3e7)
    config.history_length = kwargs['frame_stack']
    task_fn = lambda log_dir: PixelAtari(game, frame_skip=4, history_length=config.history_length,
                                         log_dir=log_dir, random_skip=kwargs['random_skip'])
    config.evaluation_env = task_fn(kwargs['log_dir'])
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=None, single_process=True)
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
    config.logger = get_logger(file_name=qr_dqn_pixel_atari.__name__)
    config.num_quantiles = 200
    config.merge(kwargs)
    run_iterations(NStepQRDQNAgent(config))

def option_qr_dqn_pixel_atari(game, **kwargs):
    config = Config()
    kwargs.setdefault('tag', option_qr_dqn_pixel_atari.__name__)
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('mean_option', 1)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('random_skip', 0)
    kwargs.setdefault('frame_stack', 4)
    config.history_length = kwargs['frame_stack']
    kwargs.setdefault('max_steps', 3e7)
    task_fn = lambda log_dir: PixelAtari(game, frame_skip=4, history_length=config.history_length,
                                         log_dir=log_dir, random_skip=kwargs['random_skip'])
    config.num_workers = 16
    config.evaluation_env = task_fn(kwargs['log_dir'])
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=None, single_process=True)
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
    config.logger = get_logger(file_name=option_qr_dqn_pixel_atari.__name__)
    config.num_quantiles = 200
    config.merge(kwargs)
    run_iterations(OptionNStepQRDQNAgent(config))

def single_run(run, game, fn, tag, **kwargs):
    np.random.seed()
    torch.manual_seed(np.random.randint(1000))
    log_dir = './log/dist_rl-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
    fn(game=game, log_dir=log_dir, tag=tag, **kwargs)

def multi_runs(game, fn, tag, **kwargs):
    kwargs.setdefault('runs', 2)
    runs = np.arange(0, kwargs['runs'])
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

def visualize(game, **kwargs):
    from skimage import io
    import json
    config = Config()
    kwargs.setdefault('tag', option_qr_dqn_pixel_atari.__name__)
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('gpu', -1)
    kwargs.setdefault('mean_option', 1)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('random_skip', 0)
    kwargs.setdefault('frame_stack', 4)
    config.history_length = kwargs['frame_stack']
    task_fn = lambda log_dir: PixelAtari(game, frame_skip=4, history_length=config.history_length,
                                         log_dir=None, random_skip=kwargs['random_skip'])
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=None, single_process=True)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: \
        OptionQuantileNet(action_dim, config.num_quantiles, kwargs['num_options'] + kwargs['mean_option'],
                          NatureConvBody(in_channels=config.history_length),
                          gpu=kwargs['gpu'])
    config.policy_fn = lambda: GreedyPolicy(epsilon=0, final_step=100000, min_epsilon=0)
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
    agent = OptionNStepQRDQNAgent(config)
    agent.load('data/saved-OptionNStepQRDQNAgent-%s.bin' % (game))
    task = task_fn(None)
    total_reward = 0
    steps = 0
    mkdir('data/%s' % (game))
    action_meanings = task.env.unwrapped.get_action_meanings()
    task.seed(0)
    state = task.reset()
    while True:
        frame = task.env.env.env.rgb_frame
        state = np.stack([state])
        quantile_values, pi, v_pi = agent.network.predict(config.state_normalizer(state))
        option = torch.argmax(pi, dim=1)
        option_quantiles = agent.candidate_quantile[agent.network.range(option.size(0)), option]
        if config.mean_option:
            mean_q_values = quantile_values.mean(-1).unsqueeze(-1)
            quantile_values = torch.cat([quantile_values, mean_q_values], dim=-1)
        q_values = quantile_values[agent.network.range(option.size(0)), :, option_quantiles]
        q_values = q_values.cpu().detach().numpy()
        actions = [agent.policy.sample(v) for v in q_values]

        option = np.asscalar(option.cpu().detach().numpy())
        mean_action = np.argmax(mean_q_values.cpu().detach().numpy().flatten())
        action = actions[0]
        if option == 9:
            option_str = 'mean'
        else:
            option_str = 'quantile_%s' % ((option + 1) / 10.0)
        decision_str = '%s-%s-%s' % (option_str, action_meanings[action], action_meanings[mean_action])
        io.imsave('data/%s/frame-%d-%s.png' % (game, steps, decision_str), frame)
        state, reward, done, _ = task.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    print(total_reward)

def batch_job():
    cf = Config()
    cf.add_argument('--ind1', type=int, default=0)
    cf.add_argument('--ind2', type=int, default=2)
    cf.merge()

    # games = ['FreewayNoFrameskip-v4',
    #          'PongNoFrameskip-v4']
    games = ['FreewayNoFrameskip-v4',
             'PongNoFrameskip-v4',
             'BeamRiderNoFrameskip-v4',
             'BattleZoneNoFrameskip-v4',
             'JourneyEscapeNoFrameskip-v4',
             'MsPacmanNoFrameskip-v4']
    game = games[cf.ind1]
    # gpu = gpus[cf.ind1]

    # def task1():
    #     multi_runs(game, option_qr_dqn_pixel_atari, num_options=9,
    #                tag='9_options_only', mean_option=0, parallel=False)

    def task2():
        multi_runs(game, option_qr_dqn_pixel_atari, num_options=9, tag='mean_and_9_options', parallel=False)

    def task3():
        multi_runs(game, qr_dqn_pixel_atari, tag='original_qr_dqn', parallel=False)

    tasks = [task2, task3]
    tasks[cf.ind2]()

if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    set_one_thread()
    # batch_job()

    # qr_dqn_cliff()
    # option_qr_dqn_cliff(mean_option=False)
    # option_qr_dqn_cliff(mean_option=True, num_options=5)
    # option_qr_dqn_cliff(random_option=True)

    parallel = True
    runs = 8
    multi_runs('CliffWalking', qr_dqn_cliff, tag='qr_dqn', parallel=parallel, runs=runs)
    # multi_runs('CliffWalking', option_qr_dqn_cliff, tag='mean_option_qr_dqn',
    #            mean_option=True, parallel=parallel, runs=runs)
    multi_runs('CliffWalking', option_qr_dqn_cliff, tag='pure_quantiles_option_qr_dqn',
               mean_option=False, parallel=parallel, runs=runs)
    # multi_runs('CliffWalking', option_qr_dqn_cliff, tag='random_option_qr_dqn',
    #            random_option=True, parallel=parallel, runs=runs)

    # game = 'BreakoutNoFrameskip-v4'
    # game = 'FreewayNoFrameskip-v4'
    # game = 'SeaquestNoFrameskip-v4'
    # game = 'MsPacmanNoFrameskip-v4'
    # game = 'FrostbiteNoFrameskip-v4'
    # game = 'EnduroNoFrameskip-v4'
    # game = 'JourneyEscapeNoFrameskip-v4'
    # game = 'SolarisNoFrameskip-v4'
    # game = 'TennisNoFrameskip-v4'
    # game = 'BoxingNoFrameskip-v4'
    # game = 'IceHockeyNoFrameskip-v4'
    # game = 'DoubleDunkNoFrameskip-v4'

    # game = 'FreewayNoFrameskip-v4'
    # game = 'PongNoFrameskip-v4'

    # game = 'SkiingNoFrameskip-v4'
    # game = 'SpaceInvadersNoFrameskip-v4'
    # game = 'QbertNoFrameskip-v4'
    # game = 'DemonAttackNoFrameskip-v4'
    # game = 'BeamRiderNoFrameskip-v4'
    # game = 'UpNDownNoFrameskip-v4'

    # game = 'BattleZoneNoFrameskip-v4'
    # game = 'BankHeistNoFrameskip-v4'
    # game = 'RobotankNoFrameskip-v4'

    # games = [spec.id for spec in gym.envs.registry.all()]
    # games = [game]
    # for game in games:
    #     try:
    #         env = gym.make(game)
    #         print(env.action_space.n)
            # if env.action_space.n <= 6:
            #     print(game)
        # except:
        #     continue
    # visualize(game, num_options=9)

    # option_qr_dqn_cart_pole()
    # qr_dqn_cart_pole()

    # n_step_dqn_pixel_atari(game)
    # qr_dqn_pixel_atari(game)

    # quantile_regression_dqn_pixel_atari(game, gpu=0, tag='original_qr_dqn')
    # option_quantile_regression_dqn_pixel_atari(game, num_options=9, gpu=0, tag='mean_and_9_options')

    # qr_dqn_pixel_atari(game, gpu=0, tag='qr_dqn')
    # option_qr_dqn_pixel_atari(game, num_options=9, gpu=0, tag='%s-option-qr' % (game))

    # option_qr_dqn_pixel_atari(game, num_options=20, gpu=0, tag='option_qr_20_options')
    # option_qr_dqn_pixel_atari(game, num_options=5, gpu=1, tag='option_qr_5_options')

    # multi_runs(game, option_qr_dqn_pixel_atari, num_options=9, gpu=0, tag='mean_and_9_options', parallel=False, max_steps=int(1e8))
    # multi_runs(game, qr_dqn_pixel_atari, gpu=0, tag='original_qr_dqn', parallel=False, max_steps=int(1e8))

    # multi_runs(game, option_qr_dqn_pixel_atari, num_options=9, gpu=0, tag='mean_and_9_options', parallel=False)
    # multi_runs(game, qr_dqn_pixel_atari, gpu=0, tag='original_qr_dqn', parallel=False)