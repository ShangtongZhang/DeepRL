from examples import *
import random
from concurrent import futures
import itertools

def batch_atari():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'BreakoutNoFrameskip-v4',
        # 'AlienNoFrameskip-v4',
        # 'DemonAttackNoFrameskip-v4',
        # 'SeaquestNoFrameskip-v4',
        # 'MsPacmanNoFrameskip-v4'
    ]

    algos = [
        dqn_pixel,
        quantile_regression_dqn_pixel,
        categorical_dqn_pixel,
        a2c_pixel,
        n_step_dqn_pixel,
        option_critic_pixel,
        ppo_pixel,
    ]

    algo = algos[cf.i]

    for game in games:
        for r in range(1):
            algo(game=game, run=r, remark=algo.__name__)

    exit()


def batch_mujoco():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
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
        'HalfCheetah-v2',
        'Swimmer-v2',
        'Hopper-v2',
        'Walker2d-v2'
    ]

    params = []

    # for game in games:
    #     for noise in [0.1, 0.5, 0.9]:
    #         params.append([generate_data, dict(game=game, noise=noise)])


    def get_hps(algo):
        if algo == 'GradientDICE':
            return HyperParameters(OrderedDict(
                lr=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
                lam=[0.1, 1, 10],
            ))
        else:
            return HyperParameters(OrderedDict(
                lr=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
            ))

    # for r in range(0, 10):
    # for r in range(10, 20):
    for r in range(20, 30):
        for game in games:
            for noise in [0.1, 0.5, 0.9]:
                for algo in ['GradientDICE', 'FQE-target', 'GQ1', 'GQ2']:
                    for hp in get_hps(algo):
                        params.append([neural_ope, dict(game=game, algo=algo, run=r, hp=hp, noise=noise, dataset='%s-%s' % (game, noise))])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_boyans_chain():
    cf = Config()
    cf.add_argument('--i', type=int, default=1000)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'BoyansChainLinear-v0',
    ]
    params = []

    hps = HyperParameters(OrderedDict(lr=np.power(2.0, np.arange(-20, 0)),
                                      ridge=[0, 0.01, 0.1]))

    # # for r in range(0, 10):
    # # for r in range(10, 20):
    # for r in range(20, 30):
    #     for game in games:
    #         for pi0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #             for mu0 in ([0.5] if pi0 == 0.5 else np.round([1 - pi0, 0.5, pi0], 1)):
    #                 for algo in ['GradientDICE', 'FQE', 'GQ1', 'GQ2']:
    #                     for hp in hps:
    #                         params.append([linear_ope_boyans_chain,
    #                                        dict(game=game, algo=algo, run=r, pi0=pi0, mu0=mu0, hp=hp)])

    hps = HyperParameters(OrderedDict(lr=np.power(2.0, np.arange(-20, 0)),
                                      ridge=[0, 0.01, 0.1],
                                      lam=[0, 0.1, 1, 10]))
    # for r in range(0, 10):
    # for r in range(10, 20):
    for r in range(20, 30):
        for game in games:
            for pi0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for mu0 in ([0.5] if pi0 == 0.5 else np.round([1 - pi0, 0.5, pi0], 1)):
                    for algo in ['GradientDICE']:
                        for hp in hps:
                            params.append([linear_ope_boyans_chain,
                                           dict(game=game, algo=algo, run=r, pi0=pi0, mu0=mu0, hp=hp)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def linear_ope_boyans_chain(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('hp', None)
    if kwargs['hp'] is not None:
        kwargs.update(kwargs['hp'].dict())
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('discount', 1)
    kwargs.setdefault('lr', 0.001)
    kwargs.setdefault('max_steps', int(5e3))
    kwargs.setdefault('ridge', 0)
    kwargs.setdefault('algo', None)
    kwargs.setdefault('pi0', 0.1)
    config = Config()
    config.merge(kwargs)

    config.repr = 'linear'

    config.num_workers = 1
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr)
    sa_dim = config.state_dim + 2
    if config.algo == 'GenDICE':
        config.network_fn = lambda: LinearDICENet(sa_dim, 'squared')
    elif config.algo == 'GradientDICE':
        config.network_fn = lambda: LinearDICENet(sa_dim, 'linear')
    elif config.algo in ['FQE', 'GQ1']:
        config.network_fn = lambda: LinearGQ1Net(sa_dim)
    elif config.algo == 'GQ2':
        config.network_fn = lambda: LinearGQ2Net(sa_dim)
    else:
        raise NotImplementedError
    config.eval_interval = config.max_steps // 100
    run_steps(LinearOPEAgent(config))


def generate_data(**kwargs):
    kwargs.setdefault('game', 'Reacher-v2')
    kwargs.setdefault('noise', 0.1)
    kwargs.setdefault('rate_samples', int(1e6))
    kwargs.setdefault('data_samples', int(1e6))
    kwargs.setdefault('dataset', '%s-%s' % (kwargs['game'], kwargs['noise']))
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    NeuralOPEDataGenerator(config).collect_data()
    exit(0)


def neural_ope(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('hp', None)
    if kwargs['hp'] is not None:
        kwargs.update(kwargs['hp'].dict())
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lam', 0.1)
    kwargs.setdefault('discount', 1)
    kwargs.setdefault('lr', 0.001)
    kwargs.setdefault('max_steps', int(1e3))
    kwargs.setdefault('ridge', 0)
    kwargs.setdefault('algo', None)
    kwargs.setdefault('batch_size', 100)
    config = Config()
    config.merge(kwargs)

    config.target_network_update_freq = 100
    config.task_fn = lambda: Task(config.game)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr, weight_decay=config.ridge)
    sa_dim = config.state_dim + config.action_dim
    if config.algo == 'GradientDICE':
        config.network_fn = lambda: NeuralDICENet(
            body_tau_fn=lambda: FCBody(sa_dim, gate=F.relu),
            body_f_fn=lambda: FCBody(sa_dim, gate=F.relu)
        )
    elif config.algo in ['FQE', 'GQ1', 'GQ2', 'FQE-target']:
        config.network_fn = lambda: NeuralGQNet(
            body_nu_fn=lambda: FCBody(sa_dim, gate=F.relu),
            body_v_fn=lambda: FCBody(sa_dim, gate=F.relu)
        )
    else:
        raise NotImplementedError
    config.eval_interval = config.max_steps // 100
    run_steps(NeuralOPEAgent(config))


def counter_example_simulate():
    alpha = 0.1
    beta = 0.1
    G = np.asarray([
        [1 + alpha, -alpha],
        [beta, 1 - beta]
    ])
    eta = 1.1
    m = 10
    G2 = np.asarray([
        [1 + alpha * (m - 1), -alpha],
        [alpha * eta * (m - 1), 1 - eta * alpha]
    ])
    # print((G2 - np.eye(2)) / alpha)
    # print(1 - eta * alpha + alpha * (m - 1))
    # print(G - G2)
    # print(np.linalg.eigvals(G2 - np.eye(2)))
    # print(np.linalg.eigvals(G2))
    # G = np.asarray([
    #     [1 + 0.1 * 1.9 * alpha, - 0.1 * alpha],
    #     [1.9 * beta, 1 - beta]
    # ])

    u = np.asarray([
        [10],
        [0]
    ])
    pG = np.eye(2)
    for _ in range(100):
        pG = pG @ G
        print(pG)
        # print(G @ u)
    #     G = G @ G
    #     print(np.linalg.eigvals(G))
    exit(0)


def sample_from_simplex(dim):
    probs = np.random.rand(dim - 1)
    comp_probs = 1 - probs
    row = []
    for i, prob in enumerate(probs):
        if not i:
            row.append(probs[i])
        else:
            row.append(np.prod(comp_probs[:i]) * probs[i])
    row.append(np.prod(comp_probs))
    random.shuffle(row)
    row = np.asarray(row)
    assert np.sum(row <= 0) == 0
    assert np.sum(row >= 1) == 0
    assert np.abs(np.sum(row) - 1) < 1e-5
    return row


def compute_ergodic_dist(P):
    n = P.shape[0]
    LHS = np.transpose(P) - np.eye(n)
    LHS = np.concatenate([LHS, np.ones((1, n))], axis=0)
    RHS = np.zeros(n + 1)
    RHS[-1] = 1
    d = np.linalg.lstsq(LHS, RHS)[0]
    d = np.reshape(d, (-1, 1))
    assert np.abs(np.sum(d) - 1) < 1e-5
    assert np.sum(np.abs(np.transpose(d) @ P - np.transpose(d))) < 1e-5
    return d


def is_pd(A):
    M = np.array(A)
    return np.all(np.linalg.eigvals(M + M.transpose()) > 0)


def is_psd(A):
    M = np.array(A)
    return np.all(np.linalg.eigvals(M + M.transpose()) >= 0)


def gq2_condition_simulate(param):
    noise, xi, dim = param
    n = 10000
    counter = 0.0
    total = 0.0
    while total < n:
        P = torch.tensor([sample_from_simplex(dim) for _ in range(dim)]).float()
        d_P = torch.tensor(compute_ergodic_dist(P.numpy())).flatten().float()
        K = np.random.randint(1, dim)
        X = torch.randn((dim, K))
        eps = torch.randn(d_P.size()) * noise
        D = d_P + eps
        D = D / D.sum()
        if (D <= 0).sum() or (D >= 1).sum():
            D = torch.softmax(D, dim=0)
        D = torch.diag(D).float()
        DP = D @ P
        PTD = P.t() @ D
        F1 = torch.cat([X.t() @ D @ X, X.t() @ DP @ X], dim=1)
        F2 = torch.cat([X.t() @ PTD @ X, xi ** 2 * X.t() @ D @ X], dim=1)
        F = torch.cat([F1, F2], dim=0)
        if is_psd(F):
            counter += 1
        total += 1
    prob = counter / total
    return {
        (noise, xi, dim): prob
    }


def gq2_condition_table():
    noises = [0, 0.001, 0.01, 0.1, 1]
    dims = [5, 10, 50, 100]
    xis = [0.9, 0.99]
    results = {}
    with futures.ProcessPoolExecutor(max_workers=8) as pool:
        for res in pool.map(gq2_condition_simulate, itertools.product(noises, xis, dims)):
            results.update(res)
    for xi in xis:
        for dim in dims:
            info = '&'.join(['%.2f' % results[noise, xi, dim] for noise in noises])
            print('$\\nsa=%d$ & %s \\\\\\hline' % (dim, info))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    set_one_thread()
    random_seed()

    # gq2_condition_simulate(noise=0.01, xi=0.99, dim=10)
    gq2_condition_table()
    select_device(-1)
    # batch_boyans_chain()
    # batch_mujoco()

    # select_device(0)
    # batch_atari()

    game = 'HalfCheetah-v2'
    # game = 'Reacher-v2'
    # game = 'Walker2d-v2'
    # game = 'Hopper-v2'
    # game = 'Swimmer-v2'

    # generate_data(
    #     game=game,
    #     data_samples=int(1e6),
    #     rate_samples=int(1e6),
    # )

    # neural_ope(
    #     game=game,
    #     # algo='GradientDICE',
    #     # algo='FQE',
    #     # algo='FQE-target',
    #     algo='GQ1',
    #     # algo='GQ2',
    #     dataset='%s-0.1' % (game),
    #     lr=5e-3,
    # )

    # linear_ope_boyans_chain(
    #     game='BoyansChainLinear-v0',
    #     # algo='GenDICE',
    #     # algo='GradientDICE',
    #     # algo='FQE',
    #     # algo='GQ1',
    #     algo='GQ2',
    #     # ridge=0.001,
    #     lr=4 ** -5,
    #     log_level=0,
    #     pi0=0.1,
    #     mu0=0.9,
    # )
