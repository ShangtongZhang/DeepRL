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

def larger_ddpg_continuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', larger_ddpg_continuous.__name__)
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
        actor_body=FCBody(state_dim, (800, 600), gate=config.gate),
        critic_body=TwoLayerFCBodyWithAction(
            state_dim, action_dim, (800, 600), gate=config.gate),
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

def ddpg_shared(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', ddpg_shared.__name__)
    kwargs.setdefault('q_l2_weight', 0)
    kwargs.setdefault('reward_scale', 1.0)
    kwargs.setdefault('noise', OrnsteinUhlenbeckProcess)
    kwargs.setdefault('std', LinearSchedule(0.2))
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])

    config.task_fn = lambda **kwargs: Roboschool(game, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: DeterministicActorCriticNet(
        state_dim, action_dim,
        phi_body=FCBody(state_dim, (400, ), gate=config.gate),
        actor_body=FCBody(400, (300, ), gate=config.gate),
        critic_body=FCBodyWithAction(400, action_dim, (300, ), gate=config.gate),
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

def plan_ddpg(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', plan_ddpg.__name__)
    kwargs.setdefault('reward_scale', 1.0)
    kwargs.setdefault('noise', OrnsteinUhlenbeckProcess)
    kwargs.setdefault('std', LinearSchedule(0.2))
    kwargs.setdefault('depth', 2)
    kwargs.setdefault('critic_loss_weight', 10)
    kwargs.setdefault('num_actors', 5)
    kwargs.setdefault('detach_action', True)
    kwargs.setdefault('mask', False)
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])

    config.task_fn = lambda **kwargs: Roboschool(game, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: PlanEnsembleDeterministicNet(
        state_dim, action_dim,
        phi_body=FCBody(state_dim, (400, ), gate=F.tanh),
        num_actors=config.num_actors,
        discount=config.discount,
        detach_action=config.detach_action)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)

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
    run_episodes(PlanDDPGAgent(config))

def naive_model_ddpg(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', naive_model_ddpg.__name__)
    kwargs.setdefault('reward_scale', 1.0)
    kwargs.setdefault('noise', OrnsteinUhlenbeckProcess)
    kwargs.setdefault('std', LinearSchedule(0.2))
    kwargs.setdefault('depth', 2)
    kwargs.setdefault('critic_loss_weight', 10)
    kwargs.setdefault('num_actors', 5)
    kwargs.setdefault('detach_action', True)
    kwargs.setdefault('mask', False)
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])

    config.task_fn = lambda **kwargs: Roboschool(game, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: NaiveModelDDPGNet(
        state_dim, action_dim,
        phi_body=FCBody(state_dim, (400, ), gate=F.tanh),
        num_actors=config.num_actors,
        discount=config.discount,
        detach_action=config.detach_action)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)

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
    run_episodes(NaiveModelDDPGAgent(config))

def single_run(run, game, fn, tag, **kwargs):
    random_seed()
    log_dir = './log/plan-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
    # log_dir = './log/baseline-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
    fn(game, log_dir, tag=tag, **kwargs)

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

    # games = ['RoboschoolAnt-v1',
    #          'RoboschoolHalfCheetah-v1',
    #          'RoboschoolHopper-v1',
    #          'RoboschoolInvertedDoublePendulum-v1',
    #          'RoboschoolReacher-v1',
    #          'RoboschoolWalker2d-v1',
    #          'RoboschoolInvertedPendulumSwingup-v1']

    # games = ['Walker2DBulletEnv-v0',
    #          'AntBulletEnv-v0',
    #          'HopperBulletEnv-v0',
    #          'RacecarBulletEnv-v0',
    #          'KukaBulletEnv-v0',
    #          'MinitaurBulletEnv-v0']

    # games = [
    #     'RoboschoolAnt-v1',
    #     'RoboschoolHopper-v1',
    #     'RoboschoolWalker2d-v1',
    #     'RoboschoolHalfCheetah-v1',
    #     'RoboschoolReacher-v1',
    #     'RoboschoolHumanoid-v1'
    # ]
    # game = games[cf.ind1]

    games = [
        'RoboschoolAnt-v1',
        'RoboschoolWalker2d-v1',
        'RoboschoolHopper-v1',
        'RoboschoolHalfCheetah-v1',
        'RoboschoolReacher-v1',
        'RoboschoolHumanoid-v1',
        'RoboschoolPong-v1',
        'RoboschoolHumanoidFlagrun-v1',
        'RoboschoolHumanoidFlagrunHarder-v1',
        'RoboschoolInvertedPendulum-v1',
        'RoboschoolInvertedPendulumSwingup-v1',
        'RoboschoolInvertedDoublePendulum-v1',
    ]
    game = games[cf.ind1]

    parallel = True
    # multi_runs(game, larger_ddpg_continuous, tag='larger_ddpg', parallel=parallel)
    multi_runs(game, naive_model_ddpg, tag='naive_model',
               depth=2, parallel=parallel)
    # def task1():
    #     multi_runs(game, ddpg_shared, tag='ddpg_shared', parallel=parallel)
    #
    # def task2():
    #     multi_runs(game, plan_ddpg, tag='d1m0', parallel=parallel,
    #                depth=1, mask=False)
    #
    # def task3():
    #     multi_runs(game, plan_ddpg, tag='d2m0', parallel=parallel,
    #                depth=2, mask=False)
    #
    # task1()
    # task2()
    # task3()
    # tasks = [task1, task2, task3]
    # tasks[cf.ind2]()

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

    # naive_model_ddpg(game, depth=2)
    # plan_ddpg(game, depth=3)
    # ddpg_shared(game)

    # multi_runs(game, naive_model_ddpg, tag='naive_model', parallel=True)

    games = [
        # 'RoboschoolHopper-v1',
        'RoboschoolAnt-v1',
        'RoboschoolWalker2d-v1',
        'RoboschoolHalfCheetah-v1',
        'RoboschoolReacher-v1',
        'RoboschoolHumanoid-v1',
        'RoboschoolPong-v1',
        'RoboschoolHumanoidFlagrun-v1',
        'RoboschoolHumanoidFlagrunHarder-v1',
        'RoboschoolInvertedPendulum-v1',
        'RoboschoolInvertedPendulumSwingup-v1',
        'RoboschoolInvertedDoublePendulum-v1',
    ]
    # for game in games:
    #     multi_runs(game, ddpg_continuous, tag='baseline_ddpg', parallel=True)
        # multi_runs(game, larger_ddpg_continuous, tag='larger_ddpg', parallel=True)


