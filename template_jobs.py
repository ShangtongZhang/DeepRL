from examples import *


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

    # games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Reacher-v2']
    # games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2']
    # games = ['Hopper-v2']
    # games = ['Reacher-v2', 'HalfCheetah-v2']
    games = ['Reacher-v2']

    # lams = dict(GradientDICE=[0.1, 1],
    #             GenDICE=[0.1, 1])

    params = []

    for game in games:
        for algo in [off_policy_evaluation]:
            for discount in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                for cor in ['GradientDICE', 'GenDICE', 'DualDICE']:
                    for lam in [0.1, 1]:
                        for lr in [1e-2, 5e-3, 1e-3]:
                            # for r in range(0, 18):
                            for r in range(18, 30):
                                params.append([algo, dict(game=game, run=r, discount=discount,
                                                          correction=cor, lr=lr, lam=lam)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_boyans_chain():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'BoyansChainTabular-v0',
        'BoyansChainLinear-v0',
    ]
    params = []

    for game in games:
        for algo in ['GenDICE', 'GradientDICE', 'DualDICE']:
            if algo == 'GenDICE':
                activation = 'squared'
            elif algo == 'GradientDICE' or algo == 'DualDICE':
                activation = 'linear'
            else:
                raise NotImplementedError
            for lr in np.power(4.0, np.arange(-6, 0)):
                for r in range(0, 30):
                    for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        params.append([gradient_dice_boyans_chain,
                                       dict(game=game, algo=algo, lr=lr, discount=gamma, lam=1, run=r,
                                            ridge=0, activation=activation)])
                    for ridge in [0, 0.001, 0.01, 0.1]:
                        params.append([gradient_dice_boyans_chain,
                                       dict(game=game, algo=algo, lr=lr, discount=1, lam=1, run=r,
                                            ridge=ridge, activation=activation)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def gradient_dice_boyans_chain(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('discount', 0.5)
    kwargs.setdefault('lr', 0.001)
    kwargs.setdefault('max_steps', int(3e4))
    kwargs.setdefault('ridge', 0)
    kwargs.setdefault('oracle_dual', False)
    config = Config()
    config.merge(kwargs)

    if config.game == 'BoyansChainTabular-v0':
        config.repr = 'tabular'
    elif config.game == 'BoyansChainLinear-v0':
        config.repr = 'linear'
    else:
        raise NotImplementedError

    config.num_workers = 1
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr)
    config.network_fn = lambda: GradientDICENet(
        config.state_dim, config.action_dim, config.activation, config.repr)
    config.eval_interval = config.max_steps // 100
    run_steps(GradientDICE(config))


def td3_correction(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('correction', 'no')
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('activation', 'squared')
    kwargs.setdefault('debug', False)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    batch_size = 100
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=batch_size)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = (batch_size if config.debug else int(1e4))
    config.target_network_mix = 5e-3

    config.dice_net_fn = lambda: GradientDICEContinuousNet(
        body_tau=FCBody(config.state_dim + config.action_dim),
        body_f=FCBody(config.state_dim + config.action_dim),
        opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        activation=config.activation
    )

    sample_init_env = Task(config.game, num_envs=batch_size)
    config.sample_init_states = lambda: sample_init_env.reset()

    run_steps(TD3CorrectionAgent(config))


def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('max_steps', int(1e6))
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(TD3Agent(config))


def off_policy_evaluation(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('correction', 'no')
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('debug', False)
    kwargs.setdefault('noise_std', 0.05)
    kwargs.setdefault('dataset', 1)
    kwargs.setdefault('discount', None)
    kwargs.setdefault('lr', 0)
    kwargs.setdefault('collect_data', False)
    kwargs.setdefault('target_network_update_freq', 1)
    config = Config()
    config.merge(kwargs)

    if config.correction in ['GradientDICE', 'DualDICE']:
        config.activation = 'linear'
        config.lam = 0.1
    elif config.correction in ['GenDICE']:
        config.activation = 'squared'
        config.lam = 1
    else:
        raise NotImplementedError

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e3)
    config.eval_interval = config.max_steps // 100

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    batch_size = 128
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=batch_size)

    config.dice_net_fn = lambda: GradientDICEContinuousNet(
        body_tau_fn=lambda: FCBody(config.state_dim + config.action_dim, gate=F.relu),
        body_f_fn=lambda: FCBody(config.state_dim + config.action_dim, gate=F.relu),
        opt_fn=lambda params: torch.optim.SGD(params, lr=config.lr),
        activation=config.activation
    )

    sample_init_env = Task(config.game, num_envs=batch_size)
    config.sample_init_states = lambda: sample_init_env.reset()

    if config.collect_data:
        OffPolicyEvaluation(config).collect_data()
    else:
        run_steps(OffPolicyEvaluation(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    set_one_thread()
    random_seed()

    select_device(-1)
    # batch_boyans_chain()
    batch_mujoco()

    # select_device(0)
    # batch_atari()

    game = 'Reacher-v2'
    # td3_continuous(
    #     game=game,
    #     max_steps=int(1e4),
    # )
    off_policy_evaluation(
        # collect_data=True,
        game=game,
        correction='GradientDICE',
        # activation='linear',
        # correction='GenDICE',
        # correction='DualDICE',
        discount=0.1,
        # discount=1,
        lr=1e-2,
        lam=1,
        target_network_update_freq=1,
    )

    # td3_correction(
    #     game=game,
    #     # correction='GradientDICE',
    #     correction='GenDICE',
    #     # correction='no',
    #     debug=True,
    # )

    # gradient_dice_boyans_chain(
    #     # game='BoyansChainTabular-v0',
    #     game='BoyansChainLinear-v0',
    #     algo='GradientDICE',
    #     # algo='GenDICE',
    #     # algo='DualDICE',
    #     # ridge=0,
    #     ridge=0.001,
    #     discount=1,
    #     # activation='squared',
    #     activation='linear',
    #     lr=4 ** -5,
    #     log_level=0,
    # )
