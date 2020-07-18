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
        # 'MsPacmanNoFrameskip-v4'
    ]

    algos = [
        dqn_pixel,
        quantile_regression_dqn_pixel,
        categorical_dqn_pixel,
        rainbow_pixel,
        a2c_pixel,
        n_step_dqn_pixel,
        option_critic_pixel,
        ppo_pixel,
    ]

    params = []

    for game in games:
        for r in range(1):
            for algo in algos:
                params.append([algo, dict(game=game, run=r, remark=algo.__name__)])
            # for n_step in [1, 2, 3]:
            #     for double_q in [True, False]:
            #         params.extend([
            #             [dqn_pixel,
            #              dict(game=game, run=r, n_step=n_step, replay_cls=PrioritizedReplay, double_q=double_q,
            #                   remark=dqn_pixel.__name__)],
                        # [rainbow_pixel,
                        #  dict(game=game, run=r, n_step=n_step, noisy_linear=False, remark=rainbow_pixel.__name__)]
                    # ])
            # params.append(
            #     [categorical_dqn_pixel, dict(game=game, run=r, remark=categorical_dqn_pixel.__name__)]),
            # params.append([dqn_pixel, dict(game=game, run=r, remark=dqn_pixel.__name__)])

    algo, param = params[cf.i]
    algo(**param)
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
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for game in games:
        if 'Humanoid' in game:
            algos = [ppo_continuous]
        else:
            algos = [ppo_continuous, ddpg_continuous, td3_continuous]
        for algo in algos:
            for r in range(5):
                params.append([algo, dict(game=game, run=r)])

    algo, param = params[cf.i]
    algo(**param, remark=algo.__name__)

    exit()


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()

    select_device(-1)
    batch_mujoco()
