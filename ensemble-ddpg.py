from deep_rl import *

def d3pg_congtinuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('tag', d3pg_congtinuous.__name__)
    kwargs.setdefault('num_actors', 1)
    config.num_workers = 8
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])
    task_fn = lambda log_dir: Roboschool(game, log_dir=log_dir)
    # task_fn = lambda log_dir: Bullet(game, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=log_dir,
                                              single_process=True)

    config.network_fn = lambda state_dim, action_dim: DeterministicActorCriticNet(
        state_dim, action_dim,
        actor_body=FCBody(state_dim, (300, 200), gate=F.tanh),
        critic_body=TwoLayerFCBodyWithAction(
            state_dim, action_dim, (400, 300), gate=F.tanh),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3)
    )

    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.max_steps = 1e7
    config.random_process_fn = lambda action_dim: GaussianProcess(
        action_dim, std_schedules=[LinearSchedule(
            0.3, 0, config.max_steps / config.num_workers) for _ in range(config.num_workers)])

    config.rollout_length = 5
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    config.merge(kwargs)
    run_iterations(D3PGAgent(config))

def d3pg_ensemble(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('tag', d3pg_ensemble.__name__)
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('off_policy_actor', False)
    kwargs.setdefault('off_policy_critic', False)
    kwargs.setdefault('random_option', False)
    config.num_workers = 8
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])
    task_fn = lambda log_dir: Roboschool(game, log_dir=log_dir)
    # task_fn = lambda log_dir: Bullet(game, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=log_dir,
                                              single_process=True)

    config.network_fn = lambda state_dim, action_dim: ThinDeterministicOptionCriticNet(
        action_dim, phi_body=DummyBody(state_dim),
        actor_body=FCBody(state_dim, (300, 200), gate=F.tanh),
        critic_body=TwoLayerFCBodyWithAction(state_dim, action_dim, (400, 300), gate=F.tanh),
        num_options=kwargs['num_options'],
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3)
    )

    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.max_steps = 1e7
    config.random_process_fn = lambda action_dim: GaussianProcess(
        action_dim, std_schedules=[LinearSchedule(
            0.3, 0, config.max_steps / config.num_workers) for _ in range(config.num_workers)])

    config.rollout_length = 5
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    config.merge(kwargs)
    run_iterations(EnsembleD3PGAgent(config))

def d3pg_option(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('tag', d3pg_option.__name__)
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('off_policy_actor', False)
    kwargs.setdefault('off_policy_critic', False)
    kwargs.setdefault('random_option', False)
    kwargs.setdefault('beta_loss_weight', 0.1)
    config.num_workers = 8
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])
    task_fn = lambda log_dir: Roboschool(game, log_dir=log_dir)
    # task_fn = lambda log_dir: Bullet(game, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=log_dir,
                                              single_process=True)

    config.network_fn = lambda state_dim, action_dim: DeterministicOptionCriticNet(
        action_dim, phi_body=DummyBody(state_dim),
        actor_body=FCBody(state_dim, (300, 200), gate=F.tanh),
        beta_body=FCBody(state_dim, (300, 200), gate=F.tanh),
        critic_body=TwoLayerFCBodyWithAction(state_dim, action_dim, (400, 300), gate=F.tanh),
        num_options=kwargs['num_options'],
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3)
    )

    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.max_steps = 1e7
    config.random_process_fn = lambda action_dim: GaussianProcess(
        action_dim, std_schedules=[LinearSchedule(
            0.3, 0, config.max_steps / config.num_workers) for _ in range(config.num_workers)])
    # config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1e6, min_epsilon=0.05)
    config.policy_fn = lambda: GreedyPolicy(epsilon=0, final_step=1e6, min_epsilon=0)

    config.rollout_length = 5
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    config.termination_regularizer = 0.01
    config.merge(kwargs)
    run_iterations(OptionD3PGAgent(config))

def ddpg_continuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', ddpg_continuous.__name__)
    kwargs.setdefault('q_l2_weight', 0)
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])

    config.task_fn = lambda **kwargs: Bullet(game, **kwargs)
    # config.task_fn = lambda **kwargs: Roboschool(game, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: DeterministicActorCriticNet(
        state_dim, action_dim,
        actor_body=FCBody(state_dim, (300, 200), gate=config.gate),
        critic_body=TwoLayerFCBodyWithAction(
            state_dim, action_dim, (400, 300), gate=config.gate),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(
            params, lr=1e-3, weight_decay=config.q_l2_weight)
        )

    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.max_steps = 1e6
    config.random_process_fn = lambda action_dim: GaussianProcess(
        (action_dim, ), [LinearSchedule(0.3, 0, 1e6)])

    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    run_episodes(DDPGAgent(config))

def ensemble_ddpg(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('tag', ensemble_ddpg.__name__)
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('off_policy_actor', False)
    kwargs.setdefault('off_policy_critic', False)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])
    config.task_fn = lambda: Roboschool(game)
    config.evaluation_env = Roboschool(game, log_dir=log_dir)
    config.network_fn = lambda state_dim, action_dim: ThinDeterministicOptionCriticNet(
        action_dim, phi_body=DummyBody(state_dim),
        actor_body=FCBody(state_dim, (300, 200), gate=F.tanh),
        critic_body=TwoLayerFCBodyWithAction(state_dim, action_dim, (400, 300), gate=F.tanh),
        num_options=kwargs['num_options'],
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3)
    )
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.max_steps = 1e6
    config.random_process_fn = lambda action_dim: GaussianProcess(
        (action_dim, ), [LinearSchedule(0.3, 0, 1e6)])
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    config.merge(kwargs)
    run_episodes(EnsembleDDPGAgent(config))

def single_run(run, game, fn, tag, **kwargs):
    log_dir = './log/ensemble-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
    fn(game, log_dir, tag=tag, **kwargs)

def multi_runs(game, fn, tag, **kwargs):
    runs = np.arange(0, 5)
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

def batch_job():
    cf = Config()
    cf.add_argument('ind_game', type=int)
    cf.add_argument('ind_task', type=int)
    cf.merge()

    # games = ['RoboschoolAnt-v1', 'RoboschoolWalker2d-v1', 'RoboschoolHalfCheetah-v1']
    # games = [
    #     'RoboschoolReacher-v1',
    #     'RoboschoolHopper-v1',
    #     'RoboschoolInvertedDoublePendulum-v1'
    # ]
    games = ['RoboschoolAnt-v1',
             'RoboschoolHalfCheetah-v1',
             'RoboschoolHopper-v1',
             'RoboschoolInvertedDoublePendulum-v1',
             'RoboschoolReacher-v1',
             'RoboschoolWalker2d-v1',
             'RoboschoolInvertedPendulumSwingup-v1']

    # games = ['Walker2DBulletEnv-v0',
    #          'AntBulletEnv-v0',
    #          'HopperBulletEnv-v0',
    #          'RacecarBulletEnv-v0',
    #          'KukaBulletEnv-v0',
    #          'MinitaurBulletEnv-v0']
    game = games[cf.ind_game]

    def task1():
        multi_runs(game, d3pg_congtinuous, tag='original_d3pg', parallel=True)
        multi_runs(game, d3pg_ensemble, tag='half_policy',
                   off_policy_actor=False, off_policy_critic=True, parallel=True)
    # def task2():
        multi_runs(game, d3pg_ensemble, tag='on_policy',
                   off_policy_actor=False, off_policy_critic=False, parallel=True)
        multi_runs(game, d3pg_ensemble, tag='off_policy',
                   off_policy_actor=True, off_policy_critic=True, parallel=True)

    # tasks = [task1, task2]
    # tasks[cf.ind_task]()
    # task()
    task1()

if __name__ == '__main__':
    mkdir('data')
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    os.system('export OMP_NUM_THREADS=1')
    os.system('export MKL_NUM_THREADS=1')
    torch.set_num_threads(1)

    # game = 'RoboschoolAnt-v1'

    # d3pg_option(game)

    # game = 'AntBulletEnv-v0'
    # game = 'Walker2DBulletEnv-v0'
    # game = 'HalfCheetahBulletEnv-v0'
    # game = 'HopperBulletEnv-v0'
    # game = 'RacecarBulletEnv-v0'
    # game = 'KukaBulletEnv-v0'
    # game = 'MinitaurBulletEnv-v0'

    # game = 'ReacherBulletEnv-v0'

    multi_runs(game, ddpg_continuous, tag='original_ddpg')

    # import pybullet_envs
    # env = gym.make(game)
    # s = env.reset()
    # image = env.render('rgb_array')
    # torchvision.utils.save_image(torch.tensor(image).transpose(2, 0), 'hello.png')

    # game = 'HalfCheetahBulletEnv-v0'
    # game = 'ReacherBulletEnv-v0'

    # ddpg_continuous(game)
    # ensemble_ddpg(game, off_policy_actor=True, off_policy_critic=True)
    # d3pg_conginuous(game)
    # d3pg_ensemble(game)

    # batch_job()

    # game = 'RoboschoolWalker2d-v1'
    # game = 'RoboschoolHalfCheetah-v1'

    # game = 'RoboschoolHopper-v1'
    # game = 'RoboschoolHumanoid-v1'
    # game = 'RoboschoolReacher-v1'

    # d3pg_ensemble(game)

    # multi_runs(game, d3pg_conginuous, tag='original_d3pg', parallel=True)
    # multi_runs(game, d3pg_ensemble, tag='off_policy',
    #            off_policy_actor=True, off_policy_critic=True, parallel=True)
    # multi_runs(game, d3pg_ensemble, tag='on_policy',
    #            off_policy_actor=False, off_policy_critic=False, parallel=True)
    # multi_runs(game, d3pg_ensemble, tag='half_policy',
    #            off_policy_actor=False, off_policy_critic=True, parallel=True)

    # d3pg_ensemble(game, off_policy_actor=False, off_policy_critic=False, random_option=True)
    # d3pg_ensemble(game, off_policy_actor=False, off_policy_critic=False)
    # d3pg_ensemble(game, off_policy_actor=True, off_policy_critic=True)
    # d3pg_ensemble(game, off_policy_actor=False, off_policy_critic=True)

    # multi_runs(game, ensemble_ddpg, tag='ensemble_half_off_policy',
    #            num_options=5, off_policy_critic=True, off_policy_actor=False, parallel=True)
    # multi_runs(game, ensemble_ddpg, tag='ensemble_off_policy',
    #            num_options=5, off_policy_critic=True, off_policy_actor=True, parallel=True)
    # multi_runs(game, ensemble_ddpg, tag='ensemble_on_policy',
    #            num_options=5, off_policy_critic=False, off_policy_actor=False, parallel=True)
    # multi_runs(game, ddpg_continuous, tag='original_ddpg', parallel=True)

    # ensemble_ddpg(game, num_options=5, off_policy=True)

    # multi_runs(game, d3pg_conginuous, tag='original_d3pg')
    # multi_runs(game, d3pg_conginuous, tag='5_actors', num_actors=5)


    # multi_runs(game, ddpg_continuous, tag='original_ddpg_tanh',
    #                 gate=F.tanh, q_l2_weight=0)
    # multi_runs(game, plan_ensemble_ddpg, tag='plan_ensemble_detach',
    #                    depth=2, num_actors=5, detach_action=True)
    # multi_runs(game, plan_ensemble_ddpg, tag='plan_ensemble_no_detach',
    #                    depth=2, num_actors=5, detach_action=False)

    # plan_ensemble_ddpg(game, tag='plan_ensemble_detach',
    #                    depth=2, num_actors=5, detach_action=True)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_no_detach',
    #                    depth=2, num_actors=5, detach_action=False)

    # multi_runs(game, ddpg_continuous, tag='original_ddpg_relu',
    #                 gate=F.relu, q_l2_weight=0)

    # multi_runs(game, ddpg_continuous, tag='original_ddpg_l2_relu',
    #                 gate=F.relu, q_l2_weight=0.01)

    # plan_ensemble_ddpg(game, tag='plan_ensemble_detach',
    #                    depth=2, num_actors=5, detach_action=True)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_no_detach',
    #                    depth=2, num_actors=5, detach_action=False)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_original',
    #                    depth=2, num_actors=5, align_next_v=False, detach_action=False)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_align_next_v',
    #                    depth=2, num_actors=5, align_next_v=True, detach_action=False)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_detach_action',
    #                    depth=2, num_actors=5, align_next_v=False, detach_action=True)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_depth_1',
    #                    depth=1, num_actors=5, align_next_v=False, detach_action=False)

    # plan_ensemble_ddpg(game, depth=2, num_actors=5)
    # d3pg_conginuous(game, num_actors=1)

    # multi_runs(game, ddpg_continuous, tag='original_ddpg')
    # multi_runs(game, ensemble_ddpg, tag='5_actors', num_actors=5)
    # multi_runs(game, plan_ensemble_ddpg, tag='5_actors')

    # ensemble_ddpg(game, num_actors=10, tag='ensemble_ddpg_run_1')

