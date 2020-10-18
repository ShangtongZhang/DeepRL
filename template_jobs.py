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


def batch_robot_tabular():
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

    games = ['RobotTabular-v0']

    params = []

    for game in games:
        for lam in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]:
            for lr in [1e-3, 5e-3, 1e-2, 5e-2]:
                for r in range(30):
                    for algo in [reverse_TD_robot]:
                        params.append([algo, dict(game=game, run=r, loss_type='td', mu=0.5, pi=0.5, lr=lr, lam=lam)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_mutation():
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

    params = []

    games = ['RobotTabular-v0']
    for mutation in [1, 2]:
        for r in range(30):
            params.append([reverse_TD_robot_tasks, dict(game=games[0], mutation=mutation, run=r)])

    mutation_metas = {1: [-1, -5, -10], 2: [0.9, 1.8, 2.7]}

    games = ['Reacher-v3']
    for mutation in [1, 2]:
        for mutation_meta in mutation_metas[mutation]:
            for r in range(30):
                params.append([continuous_reverse_TD_tasks, dict(game=games[0], mutation=mutation, mutation_meta=mutation_meta, run=r)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.eval_interval = 0

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
    agent = TD3Agent(config)
    run_steps(agent)
    agent.save('./data/reverse-rl/%s-%d' % (config.game, config.max_steps))


def get_mujoco_policy(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.eval_interval = int(1e3)
    config.eval_episodes = 10

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
    agent = TD3Agent(config)
    agent.load('./data/reverse-rl/%s-%d' % (config.game, kwargs['policy_steps']))
    # agent.eval_episodes()

    return lambda state: to_np(agent.network(tensor(state)))


def reverse_TD_robot(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('pi', 0.5)
    kwargs.setdefault('mu', 0.5)
    kwargs.setdefault('lr', 0.001)
    kwargs.setdefault('num_quantiles', 20)
    kwargs.setdefault('loss_type', 'td')
    kwargs.setdefault('lam', 0)
    kwargs.setdefault('max_steps', int(5e4))
    kwargs.setdefault('frozen', False)
    kwargs.setdefault('mutation', 0)
    config = Config()
    config.merge(kwargs)

    if config.loss_type in ['mc', 'td']:
        config.num_quantiles = 1

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.network_fn = lambda: QuantileReverseTDNet(
        repr='tabular',
        num_quantiles=config.num_quantiles,
        body_fn=lambda: DummyBody(state_dim=config.state_dim),
    )
    config.optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr)

    config.eval_interval = 1000
    config.eval_episodes = 1

    def policy(state):
        actions = [np.random.choice([0, 1], p=[config.mu, 1 - config.mu])]
        rho = [(config.pi / config.mu)  if a == 0 else (1 - config.pi) / (1 - config.mu) for a in actions]
        return actions, rho

    config.policy = policy

    agent = ReverseTDAgent(config)
    if not config.frozen:
        run_steps(agent)
    return agent


def reverse_TD_robot_tasks(**kwargs):
    supervisor = reverse_TD_robot(
        loss_type='qr',
        num_quantiles=20,
        lr=1e-2,
        mu=0.5,
        pi=0.1,
        max_steps=int(3e5),
        **kwargs,
        # max_steps=int(2e3),
    )
    agent = reverse_TD_robot(
        loss_type='qr',
        num_quantiles=20,
        lr=1e-2,
        mu=0.1,
        pi=0.1,
        max_steps=int(2e4),
        frozen=True,
        **kwargs,
    )
    agent.network.load_state_dict(supervisor.network.state_dict())
    run_steps(agent)
    exit()


def continuous_reverse_TD(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('mu', 0.5)
    kwargs.setdefault('pi', 0.5)
    kwargs.setdefault('num_quantiles', 20)
    kwargs.setdefault('loss_type', 'td')
    kwargs.setdefault('lam', 0)
    kwargs.setdefault('frozen', False)
    kwargs.setdefault('mutation_meta', None)
    config = Config()
    config.merge(kwargs)

    if config.game == 'Reacher-v3':
        policy_steps = int(2e4)
    elif config.game == 'Walker2d-v2':
        policy_steps = int(3e5)
    else:
        raise NotImplementedError

    mujoco_policy = get_mujoco_policy(**kwargs, policy_steps=policy_steps, no_log=True)

    def gaussian_policy(state):
        mean = mujoco_policy(state)
        mean = tensor(mean)
        dist_mu = torch.distributions.Normal(mean, tensor(config.mu))
        action = dist_mu.sample()
        p_mu = dist_mu.log_prob(action)
        dist_pi = torch.distributions.Normal(mean, tensor(config.pi))
        p_pi = dist_pi.log_prob(action)
        rho = p_pi.sum(-1) - p_mu.sum(-1)
        rho = rho.exp().detach()
        action = to_np(action)
        rho = to_np(rho)
        return action, rho

    config.policy = gaussian_policy

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.network_fn = lambda: QuantileReverseTDNet(
        num_quantiles=config.num_quantiles,
        repr='nn',
        body_fn=lambda: FCBody(state_dim=config.state_dim)
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 5e-3)

    config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=128)
    config.target_network_update_freq = 200
    config.warm_up = 100

    config.eval_interval = 0
    config.eval_episodes = 0

    agent = ContinuousReverseTDAgent(config)
    if not config.frozen:
        run_steps(agent)
    return agent


def continuous_reverse_TD_tasks(**kwargs):
    supervisor = continuous_reverse_TD(
        loss_type='qr',
        mu=0.5,
        pi=0.1,
        max_steps=int(5e4),
        **kwargs,
    )

    agent = continuous_reverse_TD(
        loss_type='qr',
        mu=0.1,
        pi=0.1,
        max_steps=int(2e4),
        frozen=True,
        **kwargs,
    )
    agent.network.load_state_dict(supervisor.network.state_dict())
    run_steps(agent)


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    set_one_thread()
    random_seed()

    # select_device(0)
    # batch_atari()

    select_device(-1)
    # batch_robot_tabular()
    # batch_mutation()

    # game = 'Reacher-v3'
    game = 'Walker2d-v2'
    # game = 'InvertedPendulum-v2'
    # td3_continuous(
    #     game=game,
    #     eval_episodes=0,
    #     max_steps=int(1e5),
    # )
    # continuous_reverse_TD(
    #     game=game,
    #     # loss_type='td',
    #     # loss_type='mc',
    #     loss_type='qr',
    #     num_quantiles=10,
    #     log_level=5,
    #     mu=0.5,
    #     pi=0.5,
    # )

    continuous_reverse_TD_tasks(
        game=game,
        mutation=1,
        # mutation_meta=-1,
        # mutation_meta=-5,
        mutation_meta=-10,
        # mutation=2,
        # mutation_meta=0.9,
        # mutation_meta=1.8,
        # mutation_meta=2.7,
    )

    # game = 'RobotTabular-v0'
    # reverse_TD_robot_tasks(
    #     game=game,
    #     mutation=2
    # )

    # reverse_TD_robot(
    #     game=game,
    #     lam=0,
    #     loss_type='td',
    #     # loss_type='qr',
    #     num_quantiles=20,
    #     log_level=5,
    #     lr=1e-2,
    #     mu=0.5,
    #     pi=0.5,
    #     max_steps=int(3e5),
    # )
