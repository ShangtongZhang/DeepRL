from deep_rl import *


def foo(game, **kwargs):
    kwargs.setdefault('tag', foo.__name__)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    config = Config()
    config.merge(kwargs)


def batch():
    cf = Config()
    cf.add_argument('--i1', type=int, default=0)
    cf.add_argument('--i2', type=int, default=0)
    cf.merge()

    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2']
    # games = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
    # games = ['Humanoid-v2', 'HumanoidStandup-v2']
    # games = ['RoboschoolHumanoid-v1', 'RoboschoolAnt-v1', 'RoboschoolHumanoidFlagrun-v1', 'RoboschoolHumanoidFlagrunHarder-v1']
    # games = [games[1], games[3]]
    # game = games[cf.i1]
    # games = games[4:]
    # games = [
    #     'dm-humanoid-stand',
    #     'dm-humanoid-walk',
    #     'dm-humanoid-run',
    # ]
    # algo = cf.i1 // 4
    # if algo == 0:
    # ddpg_continuous(game=game, run=cf.i2, remark='ddpg')
    # matrix_ddpg_continuous(game=game, run=cf.i2, remark='ucb', std_weight=[4, 2, 0.5, 0.125][cf.i1])
    params = [
        # dict(max_uncertainty=1, action_noise=0, random_t_mask=False),
        # dict(max_uncertainty=1, action_noise=0.05, random_t_mask=False),
        # dict(max_uncertainty=1, action_noise=0.1, random_t_mask=False),
        # dict(max_uncertainty=1, action_noise=0.2, random_t_mask=False),
        # dict(max_uncertainty=1, action_noise=0.1, random_t_mask=True),
        # dict(max_uncertainty=float('inf'), action_noise=0.1),
        # dict(max_uncertainty=float('inf'), action_noise=0),

        # dict(max_uncertainty=1, action_noise=0.1, live_action=False, plan_steps=1),
        # dict(max_uncertainty=2, action_noise=0.1, live_action=False, plan_steps=1),
        # dict(max_uncertainty=4, action_noise=0.1, live_action=False, plan_steps=1),
        # dict(max_uncertainty=1, action_noise=0.1, live_action=False, plan_steps=2),
        # dict(max_uncertainty=1, action_noise=0.1, live_action=False, plan_steps=4),
        # dict(max_uncertainty=1, action_noise=0.1, live_action=True, plan_steps=1),
        # dict(max_uncertainty=1, action_noise=0.1, live_action=False, plan_steps=1, plan_actor=True),

        # dict(max_uncertainty=2, action_noise=0.1, live_action=False, plan_steps=2, model_agg='mean'),
        # dict(max_uncertainty=2, action_noise=0.1, live_action=False, plan_steps=2, model_agg='min'),
        # dict(max_uncertainty=2, action_noise=0.1, live_action=False, plan_steps=2, model_agg='max'),

        # dict(max_uncertainty=2, action_noise=0.1, live_action=False, plan_steps=1, model_agg='mean'),
        # dict(max_uncertainty=2, action_noise=0.05, live_action=False, plan_steps=1, model_agg='mean'),
        # dict(max_uncertainty=2, action_noise=0.2, live_action=False, plan_steps=1, model_agg='mean'),
        # dict(max_uncertainty=1, action_noise=0.1, live_action=False, plan_steps=2, model_agg='mean'),

        # dict(max_uncertainty=2, action_noise=0, live_action=False, plan_steps=1, state_noise=0.05, plan_actor=True),
        # dict(max_uncertainty=2, action_noise=0, live_action=False, plan_steps=1, state_noise=0.1, plan_actor=True),
        # dict(max_uncertainty=2, action_noise=0, live_action=False, plan_steps=1, state_noise=0.2, plan_actor=True),

        # dict(max_uncertainty=2, action_noise=0, live_action=False, plan_steps=1, state_noise=0.05, plan_actor=False),
        # dict(max_uncertainty=2, action_noise=0, live_action=False, plan_steps=1, state_noise=0.1, plan_actor=False),
        # dict(max_uncertainty=2, action_noise=0, live_action=False, plan_steps=1, state_noise=0.2, plan_actor=False),
        #
        # dict(max_uncertainty=2, action_noise=0, live_action=True, plan_steps=1, state_noise=0.05, plan_actor=True),
        # dict(max_uncertainty=2, action_noise=0, live_action=True, plan_steps=1, state_noise=0.1, plan_actor=True),
        # dict(max_uncertainty=2, action_noise=0, live_action=True, plan_steps=1, state_noise=0.2, plan_actor=True),

        # dict(max_uncertainty=2, action_noise=0.1, live_action=False, plan_steps=1, model_agg='mean', plan_actor=True),
        # dict(max_uncertainty=2, action_noise=0.2, live_action=False, plan_steps=1, model_agg='mean', plan_actor=True),
        # dict(max_uncertainty=1, action_noise=0.1, live_action=False, plan_steps=2, model_agg='mean', plan_actor=True),
        # dict(max_uncertainty=1, action_noise=0.2, live_action=False, plan_steps=2, model_agg='mean', plan_actor=True),

        # dict(action_noise=0.1, live_action=False, plan_steps=1),
        # dict(action_noise=0.1, live_action=True, plan_steps=1),
        # dict(action_noise=0.1, live_action=False, plan_steps=2),
        # dict(action_noise=0.2, live_action=False, plan_steps=1),
        # dict(plan=False),

        # dict(action_noise=0.1, live_action=False, plan_steps=1),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=True, plan_actor=False),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=False, plan_actor=True),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=True, plan_actor=True),
        # dict(plan=False, real_updates=2)

        # dict(action_noise=0.1, live_action=False, plan_steps=1, prediction_noise=0.01),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, prediction_noise=0.05),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, prediction_noise=0.1),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, prediction_noise=0.2),

        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.1, target_net_residual=False),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.5, target_net_residual=False),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=1.0, target_net_residual=False),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.1, target_net_residual=True),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.5, target_net_residual=True),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=1.0, target_net_residual=True),

        # dict(residual=0.01),
        # dict(residual=0.1),
        # dict(residual=0.5),
        # dict(residual=1),

        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0, target_net_residual=False),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0, target_net_residual=True),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.05, target_net_residual=False),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.05, target_net_residual=True),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.2, target_net_residual=False),
        # dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.2, target_net_residual=True),

        # dict(action_noise=0.1, plan_steps=1, residual=0.2, target_net_residual=False, skip=False),
        # dict(action_noise=0.1, plan_steps=3, residual=0.2, target_net_residual=False, skip=False),
        # dict(action_noise=0.05, plan_steps=3, residual=0.2, target_net_residual=True, skip=False),
        # dict(action_noise=0.2, plan_steps=3, residual=0.2, target_net_residual=True, skip=False),
        # dict(action_noise=0.1, plan_steps=5, residual=0.2, target_net_residual=False, skip=False),
        # dict(action_noise=0.1, plan_steps=5, residual=0.2, target_net_residual=True, skip=False),

        # dict(action_noise=0.1, plan_steps=3, residual=0.2, target_net_residual=True),
        # dict(action_noise=0.1, plan_steps=3, residual=0.2, target_net_residual=False),

        dict(action_noise=0.1, plan_steps=1, residual=0.2, target_net_residual=False),
        dict(action_noise=0.1, plan_steps=1, residual=0),

        # dict(action_noise=0.1, plan_steps=1, residual=0.2, target_net_residual=True, skip=False),
        # dict(action_noise=0.1, plan_steps=1, residual=0, target_net_residual=True, skip=False),

        # dict(residual=0.05),
        # dict(residual=0.1),
        # dict(residual=0.2),
        # dict(residual=0.4),
        # dict(residual=0.8),
        # dict(residual=1),

        # dict(skip=False, plan=False, MVE=3),
    ]

    # ddpg_continuous(game=game, run=cf.i2, remark='ddpg')
    model_ddpg_continuous(game=games[0], run=cf.i1, **params[cf.i2], small=True)
    # oracle_ddpg_continuous(game=game, run=cf.i2, **params[cf.i1])
    # residual_ddpg_continuous(game=game, run=cf.i2, **params[0], remark='residual', target_net_residual=False)

    exit()


def dm_control_batch():
    cf = Config()
    cf.add_argument('--i1', type=int, default=0)
    cf.add_argument('--i2', type=int, default=0)
    cf.merge()

    games = [
        'dm-acrobot-swingup',
        'dm-acrobot-swingup_sparse',
        'dm-ball_in_cup-catch',
        'dm-cartpole-swingup',
        'dm-cartpole-swingup_sparse',
        'dm-cartpole-balance',
        'dm-cartpole-balance_sparse',
        'dm-cheetah-run',
        'dm-finger-turn_hard',
        'dm-finger-spin',
        'dm-finger-turn_easy',
        'dm-fish-upright',
        'dm-fish-swim',
        'dm-hopper-stand',
        'dm-hopper-hop',
        'dm-humanoid-stand',
        'dm-humanoid-walk',
        'dm-humanoid-run',
        'dm-manipulator-bring_ball',
        'dm-pendulum-swingup',
        'dm-point_mass-easy',
        'dm-reacher-easy',
        'dm-reacher-hard',
        'dm-swimmer-swimmer15',
        'dm-swimmer-swimmer6',
        'dm-walker-stand',
        'dm-walker-walk',
        'dm-walker-run',
    ]

    games = [
        'dm-walker-stand',
        'dm-walker-walk',
        'dm-walker-run',
    ]

    # residuals = [0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]

    params = []
    for game in reversed(games):
        for r in range(5):
            for delay in [0, 5, 10, 20, 40, 80]:
            # for delay in [100, 200, 400]:
                for res in [0, 0.05]:
                    params.append(dict(game=game, run=r, residual=res, delay=delay))

    params = params[90:]

    # residual_ddpg_continuous(**params[cf.i1], remark='residual', target_net_residual=True, residual=0.05)
    # residual_ddpg_continuous(**params[cf.i1], remark='residual', target_net_residual=True, residual=0)
    residual_ddpg_continuous(**params[cf.i1], remark='residual', target_net_residual=True)

    exit()


def batch_atari():
    cf = Config()
    cf.add_argument('--i1', type=int, default=0)
    cf.add_argument('--i2', type=int, default=0)
    cf.merge()

    games = [
        'BreakoutNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'DemonAttackNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4'
    ]

    params = []
    for game in games:
        for r in range(2):
            # params.append(dict(residual=0, target_net_residual=True, game=game, run=r))
            # params.append(dict(residual=0.05, target_net_residual=True, game=game, run=r))

            params.append(dict(residual=0.05, target_net_residual=True, game=game, run=r, r_aware=True))
            params.append(dict(residual=0.5, target_net_residual=True, game=game, run=r, r_aware=True))
            params.append(dict(residual=1, target_net_residual=True, game=game, run=r, r_aware=True))


            # params.append(dict(multi_step=True, entropy_weight=0.01, game=game, run=r))
            # params.append(dict(multi_step=True, entropy_weight=0.02, game=game, run=r))
            # params.append(dict(multi_step=True, entropy_weight=0.04, game=game, run=r))
            # params.append(dict(multi_step=True, entropy_weight=0.08, game=game, run=r))

            # params.append(dict(multi_step=False, residual=0, game=game, run=r, target_net_residual=True))
            # params.append(dict(multi_step=False, residual=0.05, game=game, run=r, target_net_residual=True))
            # params.append(dict(multi_step=False, residual=0.1, game=game, run=r, target_net_residual=True))
            # params.append(dict(multi_step=False, residual=0.2, game=game, run=r, target_net_residual=True))

            # params.append(dict(multi_step=False, residual=0, game=game, run=r, target_net_residual=False))
            # params.append(dict(multi_step=False, residual=0.05, game=game, run=r, target_net_residual=False))
            # params.append(dict(multi_step=False, residual=0.1, game=game, run=r, target_net_residual=False))
            # params.append(dict(multi_step=False, residual=0.2, game=game, run=r, target_net_residual=False))

    residual_dqn_pixel_atari(**params[cf.i1])
    # residual_a2c_pixel_atari(**params[cf.i1])

    exit()


def ddpg_continuous(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('gate', F.relu)
    kwargs.setdefault('weight_decay', 0)
    kwargs.setdefault('state_norm', False)
    kwargs.setdefault('skip', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(kwargs['game'])
    config.eval_env = Task(kwargs['game'], log_dir=kwargs['log_dir'])
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    if kwargs['state_norm']:
        config.state_normalizer = MeanStdNormalizer()

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=kwargs['gate']),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=kwargs['gate']),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=kwargs['weight_decay']))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.min_memory_size = int(1e4)
    config.target_network_mix = 1e-3
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])
    run_steps(DDPGAgent(config))


def model_ddpg_continuous(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('gate', F.relu)
    kwargs.setdefault('weight_decay', 0)
    kwargs.setdefault('state_norm', False)
    kwargs.setdefault('skip', True)
    kwargs.setdefault('debug', False)
    kwargs.setdefault('num_models', 1)
    kwargs.setdefault('ensemble_size', 1)
    kwargs.setdefault('bootstrap_prob', 1)
    kwargs.setdefault('plan', True)
    kwargs.setdefault('max_uncertainty', 1)
    kwargs.setdefault('plan_warm_up', int(1e4))
    kwargs.setdefault('MVE_warm_up', int(1e4))
    kwargs.setdefault('agent_warm_up', int(1e4))
    kwargs.setdefault('action_noise', 0.1)
    kwargs.setdefault('random_t_mask', False)
    kwargs.setdefault('live_action', False)
    kwargs.setdefault('plan_steps', 1)
    kwargs.setdefault('plan_actor', False)
    kwargs.setdefault('state_noise', 0)
    kwargs.setdefault('async_replay', True)
    kwargs.setdefault('residual', 0)
    kwargs.setdefault('target_net_residual', False)
    kwargs.setdefault('MVE', 0)
    kwargs.setdefault('max_steps', int(1e6))
    kwargs.setdefault('small', False)
    config = Config()
    config.merge(kwargs)

    if config.async_replay:
        replay = AsyncReplay
    else:
        replay = Replay

    config.task_fn = lambda: Task(kwargs['game'])
    config.eval_env = Task(kwargs['game'], log_dir=kwargs['log_dir'])
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    if kwargs['state_norm']:
        config.state_normalizer = MeanStdNormalizer()

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=kwargs['gate']),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=kwargs['gate']),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=kwargs['weight_decay']))

    config.replay_fn = lambda: replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))

    config.target_network_mix = 1e-3
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])

    config.model_fn = lambda: Model(config.state_dim,
                                    config.action_dim,
                                    config.ensemble_size,
                                    small=config.small)
    config.model_opt_fn = lambda params: torch.optim.Adam(params, lr=1e-3)
    config.model_replay_fn = lambda: replay(memory_size=int(1e6), batch_size=1024)
    config.model_opt_epochs = 4
    config.model_warm_up = config.agent_warm_up // 2

    if kwargs['debug']:
        config.model_warm_up = int(1e3)
        config.agent_warm_up = int(1e3)
        config.plan_warm_up = int(1e3)
        config.MVE_warm_up = int(1e3)

    run_steps(ModelDDPGAgent(config))


def backward_model_ddpg_continuous(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('gate', F.relu)
    kwargs.setdefault('weight_decay', 0)
    kwargs.setdefault('state_norm', False)
    kwargs.setdefault('skip', True)
    kwargs.setdefault('debug', False)
    kwargs.setdefault('num_models', 1)
    kwargs.setdefault('ensemble_size', 5)
    kwargs.setdefault('bootstrap_prob', 0.5)
    kwargs.setdefault('model_type', 'D')
    kwargs.setdefault('plan', True)
    kwargs.setdefault('max_uncertainty', 1)
    kwargs.setdefault('plan_warm_up', int(1e4))
    kwargs.setdefault('agent_warm_up', int(1e4))
    kwargs.setdefault('action_noise', 0)
    kwargs.setdefault('random_t_mask', False)
    kwargs.setdefault('live_action', False)
    kwargs.setdefault('plan_steps', 1)
    kwargs.setdefault('plan_actor', False)
    kwargs.setdefault('model_agg', 'mean')
    kwargs.setdefault('state_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(kwargs['game'])
    config.eval_env = Task(kwargs['game'], log_dir=kwargs['log_dir'])
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    if kwargs['state_norm']:
        config.state_normalizer = MeanStdNormalizer()

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=kwargs['gate']),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=kwargs['gate']),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=kwargs['weight_decay']))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))

    config.target_network_mix = 1e-3
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])

    config.model_fn = lambda: BackwardModel(config.state_dim,
                                            config.action_dim,
                                            128,
                                            config.ensemble_size,
                                            config.model_type)
    config.model_opt_fn = lambda params: torch.optim.Adam(params, lr=1e-3)
    config.model_replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=512, drop_prob=0)
    config.model_opt_epochs = 4

    if kwargs['debug']:
        config.agent_warm_up = int(1e3)
        config.plan_warm_up = int(1e3)

    config.model_warm_up = config.agent_warm_up // 2

    run_steps(BackwardModelDDPGAgent(config))


def oracle_ddpg_continuous(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('gate', F.relu)
    kwargs.setdefault('weight_decay', 0)
    kwargs.setdefault('state_norm', False)
    kwargs.setdefault('skip', True)
    kwargs.setdefault('debug', False)
    kwargs.setdefault('num_models', 1)
    kwargs.setdefault('ensemble_size', 5)
    kwargs.setdefault('bootstrap_prob', 0.5)
    kwargs.setdefault('model_type', 'D')
    kwargs.setdefault('plan', True)
    kwargs.setdefault('max_uncertainty', 1)
    kwargs.setdefault('plan_warm_up', int(1e4))
    kwargs.setdefault('agent_warm_up', int(1e4))
    kwargs.setdefault('action_noise', 0)
    kwargs.setdefault('random_t_mask', False)
    kwargs.setdefault('live_action', False)
    kwargs.setdefault('plan_steps', 1)
    kwargs.setdefault('plan_actor', False)
    kwargs.setdefault('model_agg', 'mean')
    kwargs.setdefault('state_noise', 0)
    kwargs.setdefault('residual', False)
    kwargs.setdefault('real_updates', 1)
    kwargs.setdefault('prediction_noise', 0)
    kwargs.setdefault('target_net_residual', False)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(kwargs['game'], auto_reset=False)
    config.eval_env = Task(kwargs['game'], log_dir=kwargs['log_dir'])
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    if kwargs['state_norm']:
        config.state_normalizer = MeanStdNormalizer()

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=kwargs['gate']),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=kwargs['gate']),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=kwargs['weight_decay']))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64, to_np=False)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))

    config.target_network_mix = 1e-3
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])

    config.model_fn = lambda: Model(config.state_dim,
                                    config.action_dim,
                                    config.ensemble_size)
    config.model_opt_fn = lambda params: torch.optim.Adam(params, lr=1e-3)
    config.model_replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=512, drop_prob=0)
    config.model_opt_epochs = 4
    config.model_warm_up = config.agent_warm_up // 2

    if kwargs['debug']:
        config.agent_warm_up = int(1e3)
        config.plan_warm_up = int(1e3)

    run_steps(OracleDDPGAgent(config))


def residual_ddpg_continuous(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('gate', F.relu)
    kwargs.setdefault('weight_decay', 0)
    kwargs.setdefault('state_norm', False)
    kwargs.setdefault('skip', True)
    kwargs.setdefault('residual', 0.1)
    kwargs.setdefault('symmetric', True)
    kwargs.setdefault('delay', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(kwargs['game'], reward_delay=config.delay)
    config.eval_env = Task(kwargs['game'], log_dir=kwargs['log_dir'])
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    if kwargs['state_norm']:
        config.state_normalizer = MeanStdNormalizer()

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=kwargs['gate']),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=kwargs['gate']),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=kwargs['weight_decay']))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.min_memory_size = int(1e4)
    config.target_network_mix = 1e-3
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])
    run_steps(ResidualDDPGAgent(config))


def residual_dqn_pixel_atari(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('skip', False)
    kwargs.setdefault('target_net_residual', True)
    kwargs.setdefault('residual', 0)
    kwargs.setdefault('debug', False)
    kwargs.setdefault('r_aware', False)

    config = Config()
    config.merge(kwargs)

    if kwargs['debug']:
        replay = Replay
    else:
        replay = AsyncReplay

    config.history_length = 4
    config.task_fn = lambda: Task(kwargs['game'], log_dir=kwargs['log_dir'])
    config.eval_env = Task(kwargs['game'], episode_life=False)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.replay_fn = lambda: replay(memory_size=int(1e6), batch_size=32)

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = False
    config.max_steps = int(2e7)
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])

    if kwargs['debug']:
        config.exploration_steps = 1000
        config.async_actor = False

    run_steps(DQNAgent(config))


def residual_a2c_pixel_atari(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('skip', False)
    kwargs.setdefault('target_net_residual', True)
    kwargs.setdefault('residual', 0)
    kwargs.setdefault('debug', False)
    kwargs.setdefault('symmetric', False)
    kwargs.setdefault('multi_step', False)
    kwargs.setdefault('entropy_weight', 0.01)

    config = Config()
    config.merge(kwargs)

    config.num_workers = 16
    config.task_fn = lambda: Task(kwargs['game'], num_envs=config.num_workers, log_dir=kwargs['log_dir'])
    config.eval_env = Task(kwargs['game'], episode_life=False)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: ResidualACNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.max_steps = int(1e7)
    config.target_network_update_freq = 10000
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])
    run_steps(ResidualA2CAgent(config))


def residual_a2c_cart_pole(**kwargs):
    kwargs.setdefault('game', 'CartPole-v0')
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('skip', False)
    kwargs.setdefault('target_net_residual', True)
    kwargs.setdefault('residual', 0.1)
    kwargs.setdefault('debug', False)
    kwargs.setdefault('symmetric', False)
    kwargs.setdefault('multi_step', False)

    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(kwargs['game'], num_envs=config.num_workers)
    config.eval_env = Task(kwargs['game'])
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: ResidualACNet(
        config.state_dim, config.action_dim,
        # FCBody(config.state_dim),
        actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim),
    )
    config.discount = 0.99
    config.logger = get_logger()
    config.entropy_weight = 0.1
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.target_network_update_freq = 800
    run_steps(ResidualA2CAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()
    # select_device(-1)
    # dm_control_batch()
    select_device(0)
    batch_atari()
    # batch()

    # game = 'HalfCheetah-v2'
    # game = 'Reacher-v2'
    # game = 'Walker2d-v2'
    game = 'Swimmer-v2'
    # game = 'RoboschoolHumanoid-v1'
    # game = 'Humanoid-v2'
    # game = 'Hopper-v2'
    # game = 'dm-cartpole-swingup'

    # residual_a2c_pixel_atari(game='BreakoutNoFrameskip-v4')

    # residual_a2c_cart_pole(
        # game='LunarLander-v2',
    # )

    # residual_ddpg_continuous(game=game,
    #                          residual=0.05,
    #                          target_net_residual=True,
    #                          symmetric=False,
    #                          delay=10)

    # ddpg_continuous(game=game)
    # backward_model_ddpg_continuous(game=game,
    #                                skip=False,
    #                                debug=False,
    #                                plan=True,
    #                                max_uncertainty=float('inf'),
    #                                action_noise=0.1,
    #                                plan_steps=1,
    #                                live_action=False,
    #                                plan_actor=True,
    #                                )

    # model_ddpg_continuous(game=game,
    #                       skip=True,
    #                       debug=True,
    #                       plan=False,
    #                       async_replay=False,
    #                       residual=0.2,
    #                       MVE=3,
    #                       small=True,
    #                       )

    game = 'BreakoutNoFrameskip-v4'
    residual_dqn_pixel_atari(game=game,
                             skip=False,
                             debug=True,
                             residual=0.05,
                             r_aware=True)

    oracle_ddpg_continuous(game=game,
                           skip=False,
                           debug=True,
                           plan=False,
                           action_noise=0.1,
                           plan_steps=2,
                           live_action=False,
                           plan_actor=True,
                           residual=0.1,
                           prediction_noise=0.1,
                           target_net_residual=True,
                           analyse=10,
                           # analyse_net='target',
                           analyse_net='online',
                           )

    # residual_ddpg_continuous(game=game,
    #                          residual=0.1,
    #                          target_net_residual=True)
