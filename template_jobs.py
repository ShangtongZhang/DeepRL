from deep_rl import *

def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
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

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
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
    run_steps(TD3Agent(config))

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

    params = []

    for game in games:
        for algo in [td3_continuous]:
            for r in range(10):
                params.append([algo, dict(game=game, run=r)])

    algo, param = params[cf.i]
    algo(**param, remark='TD3-random')

    exit()


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()

    select_device(-1)
    batch_mujoco()
