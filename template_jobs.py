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

    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Reacher-v2']

    games = ['Reacher-v2']

    params = []

    for game in games:
        for algo in ['ace', 'cof-pac']:
            for r in range(30):
                params.append([cof_pac, dict(game=game, run=r, algo=algo)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_baird():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.merge()

    params = []

    games = ['OneHotBaird-v0', 'ZeroHotBaird-v0', 'AliasedBaird-v0', 'OriginalBaird-v0']
    lr = {}
    lr[games[0]] = [2e-1, 1e-1, 5e-2]
    lr[games[1]] = [5e-2,  2.5e-2, 1.25e-2]
    lr[games[2]] = [0.1 / 32, 0.1 / 64, 0.1 / 128]
    lr[games[3]] = [0.1 / 2, 0.1 / 4, 0.1 / 8]

    lrs = 0.1 * np.power(2.0, -np.arange(1, 11))

    for game in games[2:3]:
        for r in range(30):
            # for lr_m in lr[game]:
            for lr_m in lrs:
                # for pi in [0.1, 0.3, 0.5]:
                # for pi in [0.1, 0.3]:
                # for pi in [0.5]:
                params.append([gem_baird, dict(game=game, run=r, lr_m=lr_m, pi_solid=0.1)])
                params.append([gem_baird, dict(game=game, run=r, lr_m=lr_m, pi_solid=0.3, max_steps=int(2e6))])
            # for lr_pi in [1e-2]:
            #     params.append([cofpac_baird, dict(game=game, run=r, m_type='gem', lr_pi=lr_pi, lr_m=lr_pi * 10)])
            #     params.append([cofpac_baird, dict(game=game, run=r, m_type='trace', lr_pi=lr_pi)])
            # params.append([cofpac_baird, dict(game=game, run=r, m_type='gem', lr_pi=lr_pi, lr_m=lr_pi * 2)])
            # params.append([cofpac_baird, dict(game=game, run=r, m_type='gem', lr_pi=lr_pi, lr_m=lr_pi * 4)])
            # params.append([cofpac_baird, dict(game=game, run=r, m_type='gem', lr_pi=lr_pi, lr_m=lr_pi * 8)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_baird_etd():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.merge()

    params = []

    games = ['OneHotBaird-v0', 'ZeroHotBaird-v0', 'AliasedBaird-v0', 'OriginalBaird-v0']
    # games = ['OriginalBaird-v0']
    lr = {}
    # lr[games[0]] = 0.1 * np.power(2.0, -np.arange(4, 14))
    lr[games[0]] = 0.1 * np.power(2.0, -np.arange(1, 20))

    for game in games:
        for r in range(30):
            for pi in [0.05]:
                for lr_etd in 0.1 * np.power(2.0, -np.arange(0, 20)):
                    # for m_type in ['gem', 'trace', 'oracle']:
                    for m_type in ['gem', 'trace']:
                            params.append([gem_baird, dict(game=game, run=r, lr_m=0.1 / 4, pi_solid=pi, lr_etd=lr_etd, m_type=m_type, etd=True, max_steps=int(1e5))])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def gem_baird(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('max_steps', int(1e6))
    kwargs.setdefault('etd', False)
    # kwargs.setdefault('max_steps', int(1e5))
    # kwargs.setdefault('etd', True)
    kwargs.setdefault('loss_interval', int(1e3))
    kwargs.setdefault('eval_points', 100)
    kwargs.setdefault('oracle_pi_grad', False)
    kwargs.setdefault('pi_duration', int(1e6))
    kwargs.setdefault('lr_etd', 0.1 / 16)
    kwargs.setdefault('lr_m', 0.1 / 2)
    kwargs.setdefault('m_type', 'oracle')
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = Task(config.game)
    config.mu = lambda: np.asarray([1.0 / 7, 6.0 / 7])
    config.discount = 0.99
    run_steps(GEMAgent(config))


def cof_pac(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('vm_epochs', 1)
    kwargs.setdefault('max_steps', int(1e5))
    kwargs.setdefault('eval_interval', int(1e3))
    config = Config()
    config.merge(kwargs)

    config.num_workers = 10
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.action_type = config.eval_env.action_type
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: COFPACNet(
        config.state_dim, config.action_dim,
        action_type=config.action_type,
        actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim),
        emphasis_body=FCBody(config.state_dim),
    )
    config.discount = 0.99
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=10,
                                      cat_fn=lambda x: torch.cat(x, dim=0))
    config.target_network_update_freq = 200
    config.eval_episodes = 10
    config.replay_warm_up = 100
    config.log_interval = 0
    run_steps(COFPACAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    select_device(-1)

    batch_mujoco()
    # batch_baird()
    # batch_baird_etd()

    game = 'Reacher-v2'

    cof_pac(
        game=game,
        # algo='off-pac',
        # algo='ace',
        algo='cof-pac',
    )

    # gem_baird(
    #     # game='ZeroHotBaird-v0',
    #     # game='OneHotBaird-v0',
    #     game='AliasedBaird-v0',
    #     # game='OriginalBaird-v0',
    #     lr_m=0.1 * 2 ** -2,
    #     lr_etd=0.1 / 32,
    #     # m_type='oracle',
    #     m_type='gem',
    #     # m_type='trace',
    #     max_steps=int(1e6),
    #     # log_level=1,
    #     loss_interval=1000,
    #     pi_solid=0.3,
    #     etd=True,
    # )

    # select_device(0)
    # batch_atari()

    # select_device(-1)
    # batch_mujoco()
