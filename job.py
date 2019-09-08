from deep_rl import *


def batch_parameter_study():
    cf = Config()
    cf.add_argument('--i1', type=int, default=0)
    cf.add_argument('--i2', type=int, default=0)
    cf.merge()

    game = 'HalfCheetah-v2'

    coefs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    params = []
    # for c1 in coefs:
    #     for c2 in coefs:
    #         for r in range(5):
    #             params.append(dict(algo='geoff-pac', lam1=c1, lam2=c2, gamma_hat=0.1, run=r))

    for c in coefs:
        for r in range(5):
            # params.append(dict(algo='ace', lam1=c, run=r))
            if c < 1:
                params.append(dict(algo='geoff-pac', lam1=0.7, lam2=0.6, gamma_hat=c, run=r))

    # print(len(params) // 2)
    # params = params[:302]
    # params = params[302:]
    geoff_pac(game=game, **params[cf.i1])

    exit()


def batch():
    cf = Config()
    cf.add_argument('--i1', type=int, default=0)
    cf.add_argument('--i2', type=int, default=0)
    cf.merge()

    games = ['HalfCheetah-v2', 'Reacher-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2']
    # games = ['Swimmer-v2']
    # games = ['Reacher-v2']
    params = []
    for game in games:
        for r in range(10):
            # params.append(dict(algo='off-pac', run=r, game=game))
            # params.append(dict(algo='ace', lam1=0, run=r, game=game))
            # params.append(dict(algo='geoff-pac', lam1=0.3, lam2=0.1, gamma_hat=0.2, run=r, game=game))
            # params.append(dict(algo='geoff-pac', lam1=0.3, lam2=0.1, gamma_hat=0.1, run=r, game=game))
            # params.append(dict(game=game, run=r))
            params.append(dict(algo='geoff-pac', lam1=0.7, lam2=0.6, gamma_hat=0.2, run=r, game=game))
            params.append(dict(algo='geoff-pac', lam1=0.7, lam2=0.6, gamma_hat=0.1, run=r, game=game))

    # print(len(params))
    geoff_pac(**params[cf.i1])
    # ddpg_continuous(**params[cf.i1], remark='ddpg_random')

    exit()


# DDPG baseline
def ddpg_continuous(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('gate', F.relu)
    kwargs.setdefault('state_norm', None)
    kwargs.setdefault('max_steps', int(1e6))
    kwargs.setdefault('eval_interval', int(1e4))
    kwargs.setdefault('skip', False)
    config = Config()
    config.merge(kwargs)

    if kwargs['game'] in ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2']:
        config.max_steps = int(1e6)
        config.eval_interval = int(1e3)
    elif kwargs['game'] in ['Swimmer-v2']:
        config.max_steps = int(5e6)
        config.eval_interval = int(1e3)
    elif kwargs['game'] in ['Reacher-v2']:
        config.max_steps = int(2e4)
        config.eval_interval = int(1e2)
    else:
        raise NotImplementedError

    config.task_fn = lambda: Task(kwargs['game'])
    config.eval_env = Task(kwargs['game'], log_dir=kwargs['log_dir'])
    config.eval_episodes = 10

    if kwargs['state_norm']:
        config.state_normalizer = MeanStdNormalizer()

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=kwargs['gate']),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=kwargs['gate']),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.min_memory_size = int(1e4)
    config.target_network_mix = 1e-3
    config.log_interval = 0
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])
    run_steps(DDPGAgent(config))


def random_agent():
    perf = dict()
    games = ['HalfCheetah-v2', 'Reacher-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2']
    for game in games:
        config = Config()
        config.task_fn = lambda: Task(game)
        config.eval_env = Task(game)
        config.eval_episodes = 10
        config.discount = 0.99
        config.logger = get_logger(skip=True)
        agent = RNDAgent(config)
        perf[game] = agent.eval_episodes()
    with open('data/random_agent_mujoco.bin', 'wb') as f:
        pickle.dump(perf, f)


# Geoff-PAC / ACE / Off-PAC entrance
# 'algo' in ['geoff-pac', 'off-pac', 'ace']
def geoff_pac(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('skip', False)
    kwargs.setdefault('lam1', 1)
    kwargs.setdefault('lam2', 1)
    kwargs.setdefault('gamma_hat', 0.99)
    kwargs.setdefault('vc_epochs', 1)
    kwargs.setdefault('max_steps', int(1e5))
    kwargs.setdefault('eval_interval', 100)
    kwargs.setdefault('c_coef', 1e-3)
    config = Config()
    config.merge(kwargs)

    if kwargs['game'] in ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2']:
        config.max_steps = int(1e6)
        config.eval_interval = int(1e3)
    elif kwargs['game'] in ['Swimmer-v2']:
        config.max_steps = int(5e6)
        config.eval_interval = int(1e3)
    elif kwargs['game'] in ['Reacher-v2']:
        config.max_steps = int(2e4)
        config.eval_interval = int(1e2)
    else:
        raise NotImplementedError

    config.num_workers = 10
    config.task_fn = lambda: Task(kwargs['game'], num_envs=config.num_workers)
    config.eval_env = Task(kwargs['game'], log_dir=kwargs['log_dir'])
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: GeoffPACNet(
        config.state_dim, config.action_dim,
        action_type=config.action_type,
        actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim),
        cov_shift_body=FCBody(config.state_dim),
    )
    config.discount = 0.99
    config.logger = get_logger(tag=kwargs['tag'], skip=kwargs['skip'])
    config.entropy_weight = 0
    config.gradient_clip = 0.5
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=10,
                                      cat_fn=lambda x: torch.cat(x, dim=0))
    config.target_network_update_freq = 200
    config.eval_episodes = 10
    config.replay_warm_up = 100
    config.log_interval = 0
    run_steps(GeoffPACAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()
    select_device(-1)
    batch_parameter_study()
    # batch()
    # select_device(0)

    # game = 'CartPole-v0'
    # game = 'HalfCheetah-v2'
    game = 'Reacher-v2'
    # game = 'Hopper-v2'
    # game = 'Swimmer-v2'
    # game = 'Walker2d-v2'

    # ddpg_continuous(game=game)
    random_agent()

    geoff_pac(
        game=game,
        skip=False,
        # algo='off-pac',
        # algo='ace',
        algo='geoff-pac',
        lam1=0.3,
        lam2=0.1,
        gamma_hat=0.1,
        # c_coef=1e-3,
    )