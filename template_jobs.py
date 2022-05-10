import numpy as np

from deep_rl import *


def batch_baird():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    params = []

    # hps = HyperParameters(OrderedDict(lr=0.1 * np.power(2.0, -np.arange(20))))
    # for r in range(0, 10):
    # for r in range(10, 20):
    # for r in range(20, 30):
    # for r in range(30):
        # for game in ['BairdPrediction-v0']:
        #     for n in [-1, 0, 2, 4, 8, 16, 32]:
        #         for pi_dashed in [0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1]:
        #             for hp in hps:
        #                 params.append([baird, dict(game=game, n=n, run=r, hp=hp, pi_dashed=pi_dashed)])

        # for game in ['BairdControl-v0']:
        #     for n in [-1, 0, 2, 4, 8, 16, 32]:
        #         for softmax_mu in [True, False]:
        #             for tau in [0.01, 0.1, 1]:
        #                 for hp in hps:
        #                     params.append([baird, dict(game=game, n=n, run=r, hp=hp, softmax_mu=softmax_mu, tau=tau)])
    
    hps = HyperParameters(OrderedDict(
        lr=0.1 * np.power(2.0, -np.arange(20)),
        beta=[0.1, 0.2, 0.4, 0.8]))
    # for r in range(0, 5):
    # for r in range(5, 10):
    # for r in range(10, 15):
    # for r in range(15, 20):
    # for r in range(20, 25):
    for r in range(25, 30):
    # for r in range(30):
        for game in ['BairdPrediction-v0']:
            for pi_dashed in [0, 0.02, 0.04, 0.06, 0.08, 0.1]:
                for hp in hps:
                    params.append([baird, dict(game=game, n=-1, run=r, hp=hp, pi_dashed=pi_dashed)])
        for game in ['BairdControl-v0']:
            for softmax_mu in [True, False]:
                for tau in [0.01, 0.1, 1]:
                    for hp in hps:
                        params.append([baird, dict(game=game, n=-1, run=r, hp=hp, softmax_mu=softmax_mu, tau=tau)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def baird(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('hp', None)
    if kwargs['hp'] is not None:
        kwargs.update(kwargs['hp'].dict())
    kwargs.setdefault('lr', 0.01)
    kwargs.setdefault('pi_dashed', None)
    kwargs.setdefault('log_evel', 0)
    kwargs.setdefault('n', None)
    kwargs.setdefault('beta', 0.99)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(5e5)
    config.eval_interval = 5000

    config.network_fn = lambda: BairdNet(state_dim=config.state_dim)
    config.optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr)
    config.discount = 0.99
    run_steps(VRETDAgent(config))


def batch_tile_coding():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    params = []
    game = 'CartPole-v0'

    hps = HyperParameters(OrderedDict(lr=0.1 * np.power(2.0, -np.arange(20))))
    # for r in range(0, 10):
    # for r in range(10, 20):
    for r in range(20, 30):
    # for r in range(30):
        # for n in [32, 64, 128, 256, 512, 1024]:
        for eps in [0.95]:
            # for n in [-1, 0, 2, 4, 8]:
                # for i in np.arange(len(hps)):
                    # params.append([control_with_tile_coding, dict(game=game, run=r,n=n, hp=hps[i], beta=0.99, tau=0.01, eps=eps)])
            for beta in [0.1, 0.2, 0.4, 0.8]:
                for i in np.arange(len(hps)):
                    params.append([control_with_tile_coding, dict(game=game, run=r,n=-1, hp=hps[i], beta=beta, tau=0.01, eps=eps)])

    algo, param = params[cf.i]
    algo(**param)

    exit()


def control_with_tile_coding(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('hp', None)
    if kwargs['hp'] is not None:
        kwargs.update(kwargs['hp'].dict())
    kwargs.setdefault('lr', 0.01)
    kwargs.setdefault('log_evel', 0)
    kwargs.setdefault('n', None)
    kwargs.setdefault('beta', 0.99)
    kwargs.setdefault('eps', None)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, tile_coding=True)
    config.eval_env = config.task_fn()
    config.max_steps = int(5e5)
    config.eval_interval = 5000

    config.network_fn = lambda: VanillaNet(config.action_dim, DummyBody(config.state_dim))
    config.optimizer_fn = lambda params: torch.optim.SGD(params, lr=config.lr)
    config.discount = 0.99
    run_steps(VRETDAgent(config))


def tile_coding_test():
    task = Task('CartPole-v0', tile_coding=True)
    state = task.reset()
    while True:
        action = [1] 
        next_state, reward, done, _ = task.step(action)
        print(done)


def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, tile_coding=True)
    config.eval_env = Task(config.game, tile_coding=True)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim, gate=F.tanh))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    run_steps(A2CAgent(config))


def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, tile_coding=True)
    # config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    config.history_length = 1
    config.batch_size = 10
    config.discount = 0.99
    config.max_steps = 1e5

    replay_kwargs = dict(
        memory_size=int(1e4),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length)

    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.async_actor = False
    run_steps(DQNAgent(config))


def check_n_impl(n, pi_dash):
    states = 7
    gamma = 0.99
    dist_mu = np.ones(states) / states
    D_mu = np.diag(dist_mu)
    P_pi = np.zeros_like(D_mu)
    P_pi[:, :-1] = pi_dash / 6
    P_pi[:, -1] = 1 - pi_dash
    I = np.eye(states)
    interest = np.ones((7, 1))
    f = np.linalg.inv(I - gamma * P_pi.T) @ D_mu @ interest 
    D_f = np.diag(f.flatten())
    K = 0.5 * (D_f @ (I - gamma * P_pi) + (I - gamma * P_pi.T) @ D_f)
    lam = np.linalg.eigvals(K)
    lam = np.min(lam)
    norm1 = np.linalg.norm(gamma * P_pi - I, ord=2)
    norm2 = np.linalg.matrix_power(P_pi.T, n + 1) @ f 
    norm2 = np.max(np.abs(norm2))
    exp = np.power(gamma, n+1) * norm1 * norm2 - lam
    # m_norm = np.sum(np.abs(D_mu @ m))
    # thre = lam * np.min(dist_mu) / (np.max(dist_mu) ** 2 * norm * m_norm)
    # n = np.log(thre) / np.log(gamma)
    return exp < 0

def check_n():
    for pi_dash in [0, 0.02, 0.04, 0.06, 0.08, 0.1]:
        for n in np.arange(1000):
            if check_n_impl(n, pi_dash):
                print(pi_dash, n)
                break

if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # batch_tile_coding()

    # tile_coding_test()
    select_device(-1)
    # dqn_feature(game='CartPole-v0')

    # batch_baird()

    # baird(
    #     # game='BairdPrediction-v0',
    #     # n=30,
    #     # n=-1,
    #     # n=0,
    #     # lr=0.0001,
    #     game='BairdControl-v0',
    #     softmax_mu=True,
    #     # softmax_mu=True,
    #     lr=0.05,
    #     beta=0.2,
    #     # n=-1,
    #     n=2,
    #     tau=0.01
    # )

    # control_with_tile_coding(
    #     # n=30,
    #     # n=-1,
    #     # n=0,
    #     # lr=0.0001,
    #     # game='Acrobot-v1',
    #     game='CartPole-v0',
    #     lr=0.005,
    #     beta=0.99,
    #     n=-1,
    #     # n=0,
    #     # n=10,
    #     tau=0.01,
    #     eps=0.2,
    # )

    check_n()
