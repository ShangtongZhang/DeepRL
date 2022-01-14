from deep_rl import *
import argparse
import os


def batch_ppo_oracle(cf):

    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Swimmer-v2',
        'Hopper-v2',
        # 'Reacher-v2',
        # 'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for game in games:
        for r in range(5):
            for gamma in [1]:
                params.append([mp_ppo_continuous, dict(
                    game=game, run=r, remark='mpppo', discount=gamma,
                    use_oracle_v=True, mc_n=6,
                )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def mp_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('discount', 0.99)
    kwargs.setdefault('critic_update', 'mc')
    kwargs.setdefault('gae_tau', 0.95)
    kwargs.setdefault('use_gae', True)
    kwargs.setdefault('use_oracle_v', False)
    kwargs.setdefault('mc_n', 1)
    kwargs.setdefault('bootstrap_with_oracle', False)
    kwargs.setdefault('normalized_adv', True)
    kwargs.setdefault('lr_ratio', 1)
    kwargs.setdefault('max_steps', int(2e6))
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(
            config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(
        params, 3e-4 * config.lr_ratio)
    config.critic_opt_fn = lambda params: torch.optim.Adam(
        params, 1e-3 * config.lr_ratio)
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    run_steps(MPPPOAgent(config))


if __name__ == '__main__':
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--logdir', type=str, default='.')
    cf.merge()

    set_base_log_dir(cf.logdir)

    mkdir(os.path.join(cf.logdir, 'log'))
    mkdir(os.path.join(cf.logdir, 'data'))
    random_seed()

    print(cf.i)
    print(cf.logdir)

    select_device(-1)
    batch_ppo_oracle(cf)

    game = 'HalfCheetah-v2'
    game = 'Reacher-v2'
    # game = 'Hopper-v2'
    mp_ppo_continuous(
        game=game,
        discount=0.99,
        use_oracle_v=False,
        mc_n=6,
    )
