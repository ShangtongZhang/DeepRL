from deep_rl import *


def batch():
    cf = Config()
    cf.add_argument('--i1', type=int, default=0)
    cf.add_argument('--i2', type=int, default=0)
    cf.merge()

    games = ['HalfCheetah-v2', 'Swimmer-v2', 'Reacher-v2', 'Walker2d-v2', 'Hopper-v2']
    game = games[2]

    params = [
        dict(algo='off-pac'),
        dict(algo='ace', lam1=0),
        dict(algo='ace', lam1=0.5),
        dict(algo='ace', lam1=1),
        dict(algo='geoff-pac', lam1=0, lam2=0, gamma_hat=0),
        dict(algo='geoff-pac', lam1=0.5, lam2=0, gamma_hat=0),
        dict(algo='geoff-pac', lam1=1, lam2=0, gamma_hat=0),
        dict(algo='geoff-pac', lam1=0, lam2=0, gamma_hat=0.1),
        dict(algo='geoff-pac', lam1=0, lam2=0, gamma_hat=0.5),
        dict(algo='geoff-pac', lam1=0, lam2=0, gamma_hat=1),
    ]

    geoff_pac(game=game, run=cf.i2, **params[cf.i1], skip=False)


def ddpg_continuous(name, **kwargs):
    kwargs.setdefault('tag', ddpg_continuous.__name__)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('gate', None)
    kwargs.setdefault('weight_decay', None)
    kwargs.setdefault('state_norm', None)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(name)
    config.eval_env = Task(name, log_dir=kwargs['log_dir'])
    config.max_steps = int(2e6)
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
    config.logger = get_logger(tag=kwargs['tag'])
    run_steps(DDPGAgent(config))


def geoff_pac(**kwargs):
    set_tag(kwargs)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    kwargs.setdefault('skip', True)
    kwargs.setdefault('lam1', 1)
    kwargs.setdefault('lam2', 1)
    kwargs.setdefault('gamma_hat', 0.99)
    kwargs.setdefault('vc_epochs', 1)
    config = Config()

    config.merge(kwargs)
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
    config.c_coef = 1e-3
    config.gradient_clip = 0.5
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=10,
                                      cat_fn=lambda x: torch.cat(x, dim=0))
    config.target_network_update_freq = 200
    config.eval_interval = 100
    config.eval_episodes = 10
    config.replay_warm_up = 100
    config.max_steps = int(1e5)
    run_steps(GeoffPACAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()
    select_device(-1)
    batch()
    # select_device(0)

    # game = 'CartPole-v0'
    # game = 'LunarLander-v2'
    game = 'HalfCheetah-v2'
    # game = 'Reacher-v2'

    geoff_pac(
        game=game,
        skip=False,
        # algo='off-pac',
        # algo='ace',
        algo='geoff-pac',
        lam1=0,
        lam2=0,
        gamma_hat=0.1,
    )
