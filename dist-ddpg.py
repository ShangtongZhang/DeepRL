from deep_rl import *

def ddpg_continuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', ddpg_continuous.__name__)
    kwargs.setdefault('q_l2_weight', 0)
    kwargs.setdefault('reward_scale', 1.0)
    kwargs.setdefault('option_epsilon', LinearSchedule(0))
    kwargs.setdefault('action_based_noise', True)
    kwargs.setdefault('noise', OrnsteinUhlenbeckProcess)
    kwargs.setdefault('std', LinearSchedule(0.2))
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])

    config.task_fn = lambda **kwargs: Roboschool(game, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: DeterministicActorCriticNet(
        state_dim, action_dim,
        actor_body=FCBody(state_dim, (400, 300), gate=config.gate),
        critic_body=TwoLayerFCBodyWithAction(
            state_dim, action_dim, (400, 300), gate=config.gate),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(
            params, lr=1e-3, weight_decay=config.q_l2_weight)
        )

    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.reward_normalizer = RescaleNormalizer(kwargs['reward_scale'])
    config.random_process_fn = lambda action_dim: config.noise(size=(action_dim, ), std=config.std)
    config.max_steps = 1e6
    config.evaluation_episodes_interval = int(1e4)
    config.evaluation_episodes = 20
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    run_episodes(DDPGAgent(config))

def quantile_ddpg_continuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', quantile_ddpg_continuous.__name__)
    kwargs.setdefault('q_l2_weight', 0)
    kwargs.setdefault('reward_scale', 1.0)
    kwargs.setdefault('option_epsilon', LinearSchedule(0))
    kwargs.setdefault('action_based_noise', True)
    kwargs.setdefault('noise', OrnsteinUhlenbeckProcess)
    kwargs.setdefault('std', LinearSchedule(0.2))
    kwargs.setdefault('num_quantiles', 20)
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])

    config.task_fn = lambda **kwargs: Roboschool(game, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: QuantileDDPGNet(
        state_dim, action_dim, config.num_quantiles,
        actor_body=FCBody(state_dim, (400, 300), gate=config.gate),
        critic_body=TwoLayerFCBodyWithAction(
            state_dim, action_dim, (400, 300), gate=config.gate),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(
            params, lr=1e-3, weight_decay=config.q_l2_weight),
        )

    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.reward_normalizer = RescaleNormalizer(kwargs['reward_scale'])
    config.random_process_fn = lambda action_dim: config.noise(size=(action_dim, ), std=config.std)
    config.max_steps = 1e6
    config.evaluation_episodes_interval = int(1e4)
    config.evaluation_episodes = 20
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    run_episodes(QuantileDDPGAgent(config))

def ucb_ddpg_continuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', ucb_ddpg_continuous.__name__)
    kwargs.setdefault('q_l2_weight', 0)
    kwargs.setdefault('reward_scale', 1.0)
    kwargs.setdefault('option_epsilon', LinearSchedule(0))
    kwargs.setdefault('action_based_noise', True)
    kwargs.setdefault('noise', OrnsteinUhlenbeckProcess)
    kwargs.setdefault('std', LinearSchedule(0.2))
    kwargs.setdefault('num_quantiles', 20)
    kwargs.setdefault('num_actors', 5)
    kwargs.setdefault('ucb_constant', 0)
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])

    config.task_fn = lambda **kwargs: Roboschool(game, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: QuantileEnsembleDDPGNet(
        state_dim, action_dim, config.num_quantiles, config.num_actors,
        actor_body=FCBody(state_dim, (400, 300), gate=config.gate),
        critic_body=TwoLayerFCBodyWithAction(
            state_dim, action_dim, (400, 300), gate=config.gate),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(
            params, lr=1e-3, weight_decay=config.q_l2_weight),
        )

    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.reward_normalizer = RescaleNormalizer(kwargs['reward_scale'])
    config.random_process_fn = lambda action_dim: config.noise(size=(action_dim, ), std=config.std)
    config.max_steps = 1e6
    config.evaluation_episodes_interval = int(1e4)
    config.evaluation_episodes = 20
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    run_episodes(QuantileEnsembleDDPGAgent(config))

def option_ddpg_continuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', option_ddpg_continuous.__name__)
    kwargs.setdefault('q_l2_weight', 0)
    kwargs.setdefault('reward_scale', 1.0)
    kwargs.setdefault('option_epsilon', LinearSchedule(0))
    kwargs.setdefault('action_based_noise', True)
    kwargs.setdefault('noise', OrnsteinUhlenbeckProcess)
    kwargs.setdefault('std', LinearSchedule(0.2))
    kwargs.setdefault('num_quantiles', 20)
    kwargs.setdefault('num_actors', 6)
    kwargs.setdefault('beta', 0)
    kwargs.setdefault('random_option_prob', LinearSchedule(1.0, 0, int(1e6)))
    kwargs.setdefault('detach', False)
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])

    config.task_fn = lambda **kwargs: Roboschool(game, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: QuantileOptionDDPGNet(
        state_dim, action_dim, config.num_quantiles, config.num_actors,
        actor_body=FCBody(state_dim, (400, 300), gate=config.gate),
        option_body=FCBody(state_dim, (400, 300), gate=config.gate),
        critic_body=TwoLayerFCBodyWithAction(
            state_dim, action_dim, (400, 300), gate=config.gate),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(
            params, lr=1e-3, weight_decay=config.q_l2_weight)
        )

    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.reward_normalizer = RescaleNormalizer(kwargs['reward_scale'])
    config.random_process_fn = lambda action_dim: config.noise(size=(action_dim, ), std=config.std)
    config.max_steps = 1e6
    config.evaluation_episodes_interval = int(1e4)
    config.evaluation_episodes = 20
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    run_episodes(QuantileOptionDDPGAgent(config))

def single_run(run, game, fn, tag, **kwargs):
    random_seed()
    log_dir = './log/option-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
    fn(game, log_dir, tag=tag, **kwargs)

@console
def multi_runs(game, fn, tag, **kwargs):
    kwargs.setdefault('parallel', False)
    kwargs.setdefault('runs', 5)
    runs = np.arange(0, kwargs['runs'])
    if not kwargs['parallel']:
        for run in runs:
            single_run(run, game, fn, tag, **kwargs)
        return
    ps = [mp.Process(target=single_run, args=(run, game, fn, tag), kwargs=kwargs) for run in runs]
    for p in ps:
        p.start()
        time.sleep(1)
    for p in ps: p.join()

def batch_job():
    cf = Config()
    cf.add_argument('--ind1', type=int, default=0)
    cf.add_argument('--ind2', type=int, default=0)
    cf.merge()

    games = [
        # 'RoboschoolAnt-v1',
        # 'RoboschoolWalker2d-v1',
        # 'RoboschoolHopper-v1',
        # 'RoboschoolHalfCheetah-v1',
        # 'RoboschoolReacher-v1',
        # 'RoboschoolHumanoid-v1',
        # 'RoboschoolPong-v1',
        # 'RoboschoolHumanoidFlagrun-v1',
        # 'RoboschoolHumanoidFlagrunHarder-v1',
        # 'RoboschoolInvertedPendulum-v1',
        'RoboschoolInvertedDoublePendulum-v1',
        'RoboschoolInvertedPendulumSwingup-v1',
    ]
    # game = games[cf.ind1]

    for game in games:
        parallel = True
        multi_runs(game, option_ddpg_continuous, tag='b1e0', parallel=parallel,
                   beta=1, random_option_prob=LinearSchedule(1.0, 0, int(1e6)))
        # multi_runs(game, quantile_ddpg_continuous, tag='q_ddpg', parallel=parallel)

if __name__ == '__main__':
    mkdir('data')
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    os.system('export OMP_NUM_THREADS=1')
    os.system('export MKL_NUM_THREADS=1')
    torch.set_num_threads(1)
    batch_job()

    game = 'RoboschoolAnt-v1'
    # game = 'RoboschoolWalker2d-v1'
    # game = 'RoboschoolHalfCheetah-v1'
    # game = 'RoboschoolHopper-v1'
    # game = 'RoboschoolHumanoid-v1'
    # game = 'RoboschoolHumanoidFlagrun-v1'
    # game = 'RoboschoolReacher-v1'
    # game = 'RoboschoolHumanoidFlagrunHarder-v1'

    # option_ddpg_continuous(game)
    # option_ddpg_continuous(game, detach=True)
    # quantile_ddpg_continuous(game)
    # ucb_ddpg_continuous(game)
    # ucb_ddpg_continuous(game, ucb_constant=50)

    # parallel = True
    # multi_runs(game, option_ddpg_continuous, tag='b0e0', parallel=parallel,
    #            beta=0, random_option_prob=LinearSchedule(1.0, 0, int(1e6)))
    # multi_runs(game, option_ddpg_continuous, tag='b001e0', parallel=parallel,
    #            beta=0.01, random_option_prob=LinearSchedule(1.0, 0, int(1e6)))
    # multi_runs(game, option_ddpg_continuous, tag='b01e0', parallel=parallel,
    #            beta=0.1, random_option_prob=LinearSchedule(1.0, 0, int(1e6)))
    # multi_runs(game, option_ddpg_continuous, tag='b1e0', parallel=parallel,
    #            beta=1, random_option_prob=LinearSchedule(1.0, 0, int(1e6)))

    # multi_runs(game, option_ddpg_continuous, tag='b0e0', parallel=parallel,
    #            beta=0, random_option_prob=LinearSchedule(1.0, 0, int(1e6)), id=0)
    # multi_runs(game, option_ddpg_continuous, tag='b0e1', parallel=parallel,
    #            beta=0, random_option_prob=LinearSchedule(1.0, 1.0, int(1e6)), id=0)
    # multi_runs(game, option_ddpg_continuous, tag='b01e0', parallel=parallel,
    #            beta=0, random_option_prob=LinearSchedule(1.0, 0, int(1e6)), id=1)
    # multi_runs(game, option_ddpg_continuous, tag='b01e1', parallel=parallel,
    #            beta=0, random_option_prob=LinearSchedule(1.0, 1.0, int(1e6)), id=1)

    # multi_runs(game, quantile_ddpg_continuous, tag='q_ddpg', parallel=parallel)
    # multi_runs(game, ucb_ddpg_continuous, tag='ucb_ddpg_c0', parallel=parallel)
    # multi_runs(game, ucb_ddpg_continuous, tag='ucb_ddpg_c10', parallel=parallel, ucb_constant=10)
    # multi_runs(game, ucb_ddpg_continuous, tag='ucb_ddpg_c50', parallel=parallel, ucb_constant=50)

    games = [
        # 'RoboschoolAnt-v1',
        # 'RoboschoolWalker2d-v1',
        'RoboschoolHopper-v1',
        'RoboschoolHalfCheetah-v1',
        'RoboschoolReacher-v1',
        'RoboschoolHumanoid-v1',
    ]
