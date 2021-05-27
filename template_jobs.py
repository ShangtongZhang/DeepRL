from deep_rl import *


def batch_baird():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    params = []

    for r in range(30):
        for ridge in [0, 0.01, 0.1]:
            for game in ['BairdPrediction-v0']:
                for lr_t in [0.01, 1]:
                    params.append([baird, dict(game=game, lr_t=lr_t, ridge=ridge, run=r)])
            for game in ['BairdControl-v0']:
                for softmax_mu in [True, False]:
                    for lr_t in [0.001, 1]:
                        params.append([baird, dict(game=game, lr_t=lr_t, ridge=ridge, run=r, softmax_mu=softmax_mu)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def baird(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('lr_m', 0.01)
    kwargs.setdefault('lr_t', 0.01)
    kwargs.setdefault('log_evel', 0)
    kwargs.setdefault('ridge', 0.01)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(5e5)
    config.eval_interval = 5000

    config.network_fn = lambda: BairdNet(state_dim=config.state_dim)
    config.optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr_m)
    config.discount = 0.99
    run_steps(TargetNetAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    # batch_baird()

    # select_device(0)
    # batch_atari()

    # select_device(-1)
    # batch_mujoco()
    #

    baird(
        # game='BairdPrediction-v0',
        # lr_t=0.01
        game='BairdControl-v0',
        # lr_t=0.001,
        lr_t=1,
        ridge=0.1,
        # softmax_mu=True,
        softmax_mu=False,
    )
