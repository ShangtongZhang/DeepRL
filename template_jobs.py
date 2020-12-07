from deep_rl import *


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
        'Walker2d-v2',
        'Swimmer-v2',
        'Hopper-v2',
        'Reacher-v2',
        'Ant-v2',
        'InvertedPendulum-v2',
        'InvertedDoublePendulum-v2',
    ]

    params = []

    # games = ['RiskChain-v0']
    for game in games:
        # for r in range(20, 30):
        # for r in range(10, 20):
        # for r in range(4, 10):
        for r in range(0, 4):
            for action_noise in [0.1]:
                params.append([mvpi_td3_continuous, dict(game=game, run=r, lam=0, remark='mvpi_td3', EOT_eval=100, action_noise=action_noise)])
                for lam in [0.5, 1, 2]:
                    params.append([mvpi_td3_continuous, dict(game=game, run=r, lam=lam, remark='mvpi_td3', EOT_eval=100, action_noise=action_noise)])
                    params.append([mvp_continuous, dict(game=game, run=r, lam=lam, remark='mvp', EOT_eval=100, action_noise=action_noise)])
                    params.append([tamar_continuous, dict(game=game, run=r, lam=lam, remark='tamar', EOT_eval=100, action_noise=action_noise)])
                    params.append([risk_a2c_continuous, dict(game=game, run=r, lam=lam, remark='risk', EOT_eval=100, action_noise=action_noise)])
                    params.append([var_ppo_continuous, dict(game=game, run=r, lam=lam, remark='trvo', EOT_eval=100, action_noise=action_noise)])
                # params.append([var_a2c_continuous, dict(game=game, run=r, lam=lam, remark='mva2c')])
        # for meta_prob in [0.1, 0.5, 1.0]:
            #     params.append([meta_var_ppo_continuous, dict(game=game, run=r, meta_lr=1e-3, meta_prob=meta_prob)])
            # params.append([mvp_continuous, dict(game=game, run=r, remark='mvp', EOT_eval=100)])
            # params.append([tamar_continuous, dict(game=game, run=r, remark='tamar', EOT_eval=100)])
            # params.append([risk_a2c_continuous, dict(game=game, run=r, remark='risk', EOT_eval=100)])
            # for lam in [0, 1, 2, 4, 8]:
            #     for n_samples in [10, 50, 100, 500, 1000]:
            #         params.append([off_policy_mvpi, dict(game=game, run=r, lam=lam, remark='off-policy', num_samples=n_samples)])
                # params.append([off_policy_mvpi, dict(game=game, run=r, lam=lam, use_oracle_ratio=True, remark='off-policy')])

    # for game in games:
    #     for r in range(0, 10):
    #     # for r in range(5, 10):
    #     #     params.append([ppo_continuous, dict(game=game, run=r, remark='ppo')])
    #         for meta_lr in [1e-3, 1e-2]:
    #             for meta_prob in [0.1]:
    #                 params.append([meta_var_ppo_continuous, dict(game=game, run=r, meta_lr=meta_lr, meta_prob=meta_prob)])

    # for game in games:
    #     for r in range(0, 10):
    #     # for r in range(5, 10):
    #     #     for lam in [10, 1, 0.1]:
    #         for lr in [7e-5, 7e-4]:
    #             # params.append([mvp_continuous, dict(game=game, run=r, lam=lam, remark='mvp')])
    #             # params.append([mvp_continuous, dict(game=game, run=r, lr=lr, remark='mvp')])
    #             params.append([tamar_continuous, dict(game=game, run=r, lr=lr, lam=0.1, b=10, remark='tamar')])
    #         for lam in [1, 10]:
    #             params.append([tamar_continuous, dict(game=game, run=r, lr=7e-4, lam=lam, b=10, remark='tamar')])
    #         params.append([tamar_continuous, dict(game=game, run=r, lr=7e-4, lam=0.1, b=50, remark='tamar')])

    # for game in games:
    #     for r in range(0, 10):
    #         for lr in [7e-5, 7e-4]:
    #             for lam in [0.1, 1, 10]:
    #                 params.append([risk_a2c_continuous, dict(game=game, run=r, lam=lam, lr=lr, remark='risk')])

    # params = params[0: 100]
    # params = params[100: 200]
    # params = params[200: 300]
    algo, param = params[cf.i]
    algo(**param)

    exit()


def set_max_steps(config):
    config.max_steps = int(1e6)


# TRVO
def var_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, 3e-4)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, 1e-3)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    set_max_steps(config)
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(VarPPOAgent(config))


# MVP
def mvp_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lr', 7e-4)
    kwargs.setdefault('lam', 0.1)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, action_noise=config.action_noise)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr)
    config.network_fn = lambda: MVPNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim))
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    set_max_steps(config)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    config.discount = 1
    run_steps(MVPAgent(config))


# Prashanth
def risk_a2c_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('beta', 1e-4)
    kwargs.setdefault('lam', 0.1)
    kwargs.setdefault('pi_loss_weight', 0.01)
    kwargs.setdefault('lr', 7e-5)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, action_noise=config.action_noise)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr)
    config.network_fn = lambda: RiskActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
    config.discount = 0.99
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    set_max_steps(config)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(RiskA2CAgent(config))


# Tamar
def tamar_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lr', 7e-4)
    kwargs.setdefault('lam', 0.1)
    kwargs.setdefault('b', 50)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, action_noise=config.action_noise)
    config.eval_env = Task(config.game)
    config.pi_optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr)
    config.JV_optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr * 100)
    config.network_fn = lambda: TamarNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim))
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    set_max_steps(config)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    config.discount = 1
    run_steps(TamarAgent(config))


# MVPI-TD3
def mvpi_td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lam', 0.1)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()
    set_max_steps(config)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (400, 300), gate=F.relu),
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
    run_steps(MVPITD3Agent(config))


# Offline MVPI
def off_policy_mvpi(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('algo', 'GradientDICE')
    kwargs.setdefault('repr', 'tabular')
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('dice_lam', 1)
    kwargs.setdefault('discount', 0.7)
    kwargs.setdefault('max_steps', int(2e2))
    kwargs.setdefault('activation', 'linear')
    kwargs.setdefault('q_lr', 0.01)
    kwargs.setdefault('pi_lr', 0.01)
    kwargs.setdefault('tau_lr', 0.01)
    kwargs.setdefault('use_oracle_ratio', False)
    kwargs.setdefault('use_oracle_q', False)
    kwargs.setdefault('num_samples', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 1
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.q_opt_fn = lambda params: torch.optim.SGD(params, config.q_lr)
    config.pi_opt_fn = lambda params: torch.optim.SGD(params, config.pi_lr)
    config.tau_opt_fn = lambda params: torch.optim.SGD(params, config.tau_lr)
    config.network_fn = lambda: OffPolicyMVPINet(
        config.state_dim, config.action_dim, config.activation, config.repr)
    config.replay_fn = lambda: Replay(memory_size=config.num_samples, batch_size=1)
    config.eval_interval = config.max_steps // 100
    run_steps(OffPolicyMVPI(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()

    select_device(-1)
    batch_mujoco()

    # compute_boundary()

    # off_policy_mvpi(
    #     game='RiskChain-v0',
    #     lam=12,
    #     num_samples=int(1000),
    #     tau_lr=0.01,
    #     q_lr=0.01,
    #     pi_lr=0.01,
    # )

    game = 'HalfCheetah-v2'
    # game = 'Walker2d-v2'
    # game = 'Ant-v2'
    # game = 'Reacher-v2'

    mvpi_td3_continuous(
        game=game,
        lam=0.1,
        action_noise=0.1,
        max_steps=int(1e4),
        EOT_eval=10
    )

    # var_ppo_continuous(
    #     game=game,
    #     lam=0.5,
    #     EOT_eval=100,
    # )

    # mvp_continuous(
    #     game=game,
    #     lam=0.1,
    # )

    # risk_a2c_continuous(
    #     game=game,
    #     lam=0,
    # )

    # tamar_continuous(
    #     game=game,
    #     lam=0.1,
    #     b=10,
    #     lr=1e-3,
    # )

