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

    games = ['Hopper-v2', 'HalfCheetah-v2', 'Swimmer-v2', 'Reacher-v2', 'Walker2d-v2']
    game = games[cf.i1 % 5]
    state_norm = cf.i1 // 5
    ddpg_continuous(game, gate=F.relu, weight_decay=1e-2, state_norm=state_norm,
                    tag='%s_relu_norm_%d_l2_%d' % (game, state_norm, cf.i2))
    ddpg_continuous(game, gate=F.relu, weight_decay=0, state_norm=state_norm,
                    tag='%s_relu_norm_%d_nl2_%d' % (game, state_norm, cf.i2))
    ddpg_continuous(game, gate=F.tanh, weight_decay=0, state_norm=state_norm,
                    tag='%s_tanh_norm_%d_%d' % (game, state_norm, cf.i2))


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


def off_pac_cart_pole():
    config = Config()
    game = 'CartPole-v0'
    config.num_workers = 5
    config.task_fn = lambda: Task(game, num_envs=config.num_workers)
    config.eval_env = Task(game)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: OffPACNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim))
    config.discount = 0.99
    config.logger = get_logger()
    # config.entropy_weight = 0.01
    config.entropy_weight = 0
    config.gradient_clip = 0.5
    config.eval_interval = 1000
    config.eval_episodes = 10
    run_steps(OffPACAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()
    select_device(-1)
    # batch()
    # select_device(0)

    off_pac_cart_pole()
