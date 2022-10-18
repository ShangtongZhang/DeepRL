from examples import *
from gym.envs.registration import register

def batch_chain():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    params = []

    # common_kwargs = dict(
        # lr=0.01,
        # critic_weight=10,
    # )
    # for r in range(10):
    #     for num_states in [5]:
    #     # for num_states in [6, 7]:
    #     # for num_states in [5, 6, 7]:
    #         for reg_weight in [0, 0.005, 0.01, 0.5]:
    #             params.append([off_pac, dict(
    #                 on_policy=True, 
    #                 reg_weight=reg_weight,
    #                 run=r,
    #                 num_states=num_states,
    #                 **common_kwargs)])
    #             for mu_temperature in [0, 0.1, 0.2]:
    #                 params.append([off_pac, dict(
    #                     on_policy=False, 
    #                     mu_temperature=mu_temperature,
    #                     run=r,
    #                     reg_weight=reg_weight,
    #                     num_states=num_states,
    #                     **common_kwargs)])
    
    # for r in range(0, 6):
    # for r in range(6, 12):
    # for r in range(12, 18):
    # for r in range(18, 24):
    for r in range(24, 30):
    # for r in range(30):
        for num_states in [5, 6]:
        # for num_states in [7, 8, 9]:
            # for eps_reg in np.array([0.2, 0.6, 1, 1.4, 1.8, 2, 4, 8, 16, 32]) / 16: 
            # for eps_reg in np.array([0.5]) / 16:
            for eps_reg in np.array([0.5, 1, 2, 8, 32]) / 16:
                for sac in [True, False]:
                    params.append([off_pac, dict(
                        sac=sac,
                        on_policy=False, 
                        run=r,
                        num_states=num_states,
                        eps_reg=eps_reg)])


    algo, param = params[cf.i]
    algo(**param)
    exit()


def off_pac(**kwargs):
    kwargs.setdefault('game', 'Chain-v0')
    generate_tag(kwargs)
    kwargs.setdefault('mu_temperature', 0.1)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_states', 10)
    kwargs.setdefault('beta', 0.8)
    kwargs.setdefault('discount', 0.99)
    kwargs.setdefault('full_init_support', False)
    kwargs.setdefault('on_policy', True)
    kwargs.setdefault('discount_state', True)
    kwargs.setdefault('reg', 'kl')
    kwargs.setdefault('eps_critic', 0.5+0.001)
    kwargs.setdefault('eps_actor', 0.75+0.001)
    kwargs.setdefault('eps_reg', 1.0 / 16)
    kwargs.setdefault('eps_mu', 0.9)
    kwargs.setdefault('sac', False)
    config = Config()
    config.merge(kwargs)

    register(
        id='Chain-v0',
        entry_point='deep_rl.component.envs:Chain',
        kwargs=dict(
            num_states=config.num_states,
            beta=config.beta,
            gamma=config.discount,
            full_init_support=config.full_init_support,
        )
    )

    config.task_fn = lambda: Task(config.game)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.SGD(params, 1)
    config.network_fn = lambda: OffPACNet(
        state_dim=config.eval_env.state_dim, 
        action_dim=config.eval_env.action_dim)

    config.max_steps = int(2e6)
    config.eval_interval = int(2e3)
    config.eval_episodes = 10
    run_steps(OffPACAgent(config))

if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()

    # select_device(-1)
    # batch_mujoco()

    batch_chain()

    off_pac(
        sac=True,
        on_policy=False,
        num_states=8,
        mu_temperature=0.1,
        # log_level=4,
        eps_reg=1.0 / 16,
        # lr=0.01,
        # critic_weight=10,
        # reg_weight=0.1,
    )
