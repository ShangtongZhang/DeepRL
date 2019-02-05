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

    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2']
    # game = games[cf.i1]
    game = games[1]
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

        dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.1, target_net_residual=False),
        dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.5, target_net_residual=False),
        dict(action_noise=0.1, live_action=False, plan_steps=1, residual=1.0, target_net_residual=False),

        dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.1, target_net_residual=True),
        dict(action_noise=0.1, live_action=False, plan_steps=1, residual=0.5, target_net_residual=True),
        dict(action_noise=0.1, live_action=False, plan_steps=1, residual=1.0, target_net_residual=True),
    ]

    oracle_ddpg_continuous(game=game, run=cf.i2, **params[cf.i1])

    exit()


def single_run(run, game, fn, tag, **kwargs):
    random_seed()
    log_dir = './log/dist_rl-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
    fn(game=game, log_dir=log_dir, tag=tag, **kwargs)


def multi_runs(game, fn, tag, **kwargs):
    kwargs.setdefault('runs', 2)
    runs = kwargs['runs']
    if np.isscalar(runs):
        runs = np.arange(0, runs)
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

    config.model_fn = lambda: Model(config.state_dim,
                                    config.action_dim,
                                    128,
                                    config.ensemble_size,
                                    config.model_type)
    config.model_opt_fn = lambda params: torch.optim.Adam(params, lr=1e-3)
    config.model_replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=512, drop_prob=0)
    config.model_opt_epochs = 4
    config.model_warm_up = config.agent_warm_up // 2

    if kwargs['debug']:
        config.agent_warm_up = int(1e3)
        config.plan_warm_up = int(1e3)

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

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))

    config.target_network_mix = 1e-3
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])

    config.model_fn = lambda: Model(config.state_dim,
                                    config.action_dim,
                                    128,
                                    config.ensemble_size,
                                    config.model_type)
    config.model_opt_fn = lambda params: torch.optim.Adam(params, lr=1e-3)
    config.model_replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=512, drop_prob=0)
    config.model_opt_epochs = 4
    config.model_warm_up = config.agent_warm_up // 2

    if kwargs['debug']:
        config.agent_warm_up = int(1e3)
        config.plan_warm_up = int(1e3)

    run_steps(OracleDDPGAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()
    select_device(-1)
    batch()
    # select_device(0)

    game = 'HalfCheetah-v2'
    # game = 'Reacher-v2'
    # game = 'Walker2d-v2'
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
    #                       plan=True,
    #                       max_uncertainty=2,
    #                       action_noise=0.2,
    #                       plan_steps=1,
    #                       live_action=True,
    #                       plan_actor=True,
    #                       state_noise=0
    #                       )

    oracle_ddpg_continuous(game=game,
                           skip=True,
                           debug=True,
                           plan=True,
                           action_noise=0.2,
                           plan_steps=2,
                           live_action=False,
                           plan_actor=True,
                           residual=0.1,
                           prediction_noise=0.1,
                           target_net_residual=True,
                           )