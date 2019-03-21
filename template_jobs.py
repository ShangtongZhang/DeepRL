from deep_rl import *


def batch_atari():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        # 'BreakoutNoFrameskip-v4',
        'AsterixNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4'
        # 'AlienNoFrameskip-v4',
        # 'DemonAttackNoFrameskip-v4',
        # 'SeaquestNoFrameskip-v4',
    ]

    algos = [
        IOPG_pixel,
        # OC_pixel,
        # a2c_pixel,
    ]

    params = []

    for game in games:
        for r in range(4):
            for algo in algos:
                params.append([algo, dict(game=game, run=r, remark=algo.__name__)])

    params = params[cf.i]
    params[0](**params[1])

    exit()


# def batch_mujoco():
#     cf = Config()
#     cf.add_argument('--i', type=int, default=0)
#     cf.add_argument('--j', type=int, default=0)
#     cf.merge()
#
#     games = [
#         'dm-acrobot-swingup',
#         'dm-acrobot-swingup_sparse',
#         'dm-ball_in_cup-catch',
#         'dm-cartpole-swingup',
#         'dm-cartpole-swingup_sparse',
#         'dm-cartpole-balance',
#         'dm-cartpole-balance_sparse',
#         'dm-cheetah-run',
#         'dm-finger-turn_hard',
#         'dm-finger-spin',
#         'dm-finger-turn_easy',
#         'dm-fish-upright',
#         'dm-fish-swim',
#         'dm-hopper-stand',
#         'dm-hopper-hop',
#         'dm-humanoid-stand',
#         'dm-humanoid-walk',
#         'dm-humanoid-run',
#         'dm-manipulator-bring_ball',
#         'dm-pendulum-swingup',
#         'dm-point_mass-easy',
#         'dm-reacher-easy',
#         'dm-reacher-hard',
#         'dm-swimmer-swimmer15',
#         'dm-swimmer-swimmer6',
#         'dm-walker-stand',
#         'dm-walker-walk',
#         'dm-walker-run',
#     ]
#
#     games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Reacher-v2']
#
#     params = []
#
#     for game in games:
#         for r in range(5):
#             params.append(dict(game=game, run=r))
#
#     algos = [
#         # ppo_continuous,
#         ddpg_continuous,
#     ]
#     algo = algos[cf.i // 25]
#     algo(**params[cf.i % 25], remark=algo.__name__)
#
#     exit()


def IOPG_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('skip', False)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: InterOptionPGNet(FCBody(config.state_dim), config.action_dim, num_options=2)
    config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    run_steps(InterOptionPGAgent(config))


def IOPG_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('skip', False)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: InterOptionPGNet(NatureConvBody(), config.action_dim, num_options=4)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    run_steps(InterOptionPGAgent(config))


def OC_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('skip', False)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticNet(NatureConvBody(), config.action_dim, num_options=4)
    config.random_option_prob = LinearSchedule(0.1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    run_steps(OptionCriticAgent(config))


def a2c_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('skip', False)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = False
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    run_steps(A2CAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()

    select_device(0)
    batch_atari()

    # select_device(-1)
    # batch_mujoco()

    IOPG_feature(game='CartPole-v0')
