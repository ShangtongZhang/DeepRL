from deep_rl import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

def d3pg_conginuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('tag', d3pg_conginuous.__name__)
    kwargs.setdefault('value_loss_weight', 10.0)
    kwargs.setdefault('num_actors', 3)
    config.num_workers = 5
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])
    task_fn = lambda log_dir: Roboschool(game, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=log_dir)

    config.network_fn = lambda state_dim, action_dim: EnsembleDeterministicNet(
        actor_body=FCBody(state_dim, (300, 200), gate=F.tanh),
        critic_body=TwoLayerFCBodyWithAction(state_dim, action_dim, [400, 300], gate=F.tanh),
        action_dim=action_dim, num_actors=kwargs['num_actors']
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    # config.max_steps = 1e6
    config.random_process_fn = lambda action_dim: GaussianProcess(
        action_dim, LinearSchedule(0.3, 0, 1e6))

    config.rollout_length = 5
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = Logger('./log', logger)
    config.merge(kwargs)
    run_iterations(D3PGAgent(config))

def ddpg_continuous(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('gate', F.tanh)
    kwargs.setdefault('tag', ddpg_continuous.__name__)
    kwargs.setdefault('q_l2_weight', 0)
    config.merge(kwargs)
    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])
    task_fn = lambda **kwargs: Bullet(game, **kwargs)
    config.task_fn = lambda : ProcessTask(task_fn)
    config.evaluation_env = ProcessTask(task_fn, log_dir=log_dir)
    config.actor_network_fn = lambda state_dim, action_dim: DeterministicActorNet(
        action_dim, FCBody(state_dim, (300, 200), gate=config.gate))
    config.critic_network_fn = lambda state_dim, action_dim: DeterministicCriticNet(
        TwoLayerFCBodyWithAction(state_dim, action_dim, (400, 300), gate=config.gate))
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=config.q_l2_weight)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.max_steps = 1e6
    config.random_process_fn = lambda action_dim: GaussianProcess(
        action_dim, LinearSchedule(0.3, 0, 1e6))

    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = Logger('./log', logger)
    run_episodes(DDPGAgent(config))

# def ensemble_ddpg(game, log_dir=None, **kwargs):
#     config = Config()
#     kwargs.setdefault('tag', ensemble_ddpg.__name__)
#     kwargs.setdefault('value_loss_weight', 10.0)
#     kwargs.setdefault('num_actors', 5)
#     if log_dir is None:
#         log_dir = get_default_log_dir(kwargs['tag'])
#     config.task_fn = lambda: Roboschool(game)
#     config.evaluation_env = Roboschool(game, log_dir=log_dir)
#     config.network_fn = lambda state_dim, action_dim: EnsembleDeterministicNet(
#         actor_body=FCBody(state_dim, (300, 200), gate=F.tanh),
#         critic_body=TwoLayerFCBodyWithAction(state_dim, action_dim, [400, 300], gate=F.tanh),
#         action_dim=action_dim, num_actors=kwargs['num_actors']
#     )
#     config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
#     config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
#     config.discount = 0.99
#     config.state_normalizer = RunningStatsNormalizer()
#     config.max_steps = 1e6
#     config.random_process_fn = lambda action_dim: GaussianProcess(
#         action_dim, LinearSchedule(0.3, 0, 1e6))
#     config.min_memory_size = 64
#     config.target_network_mix = 1e-3
#     config.logger = Logger('./log', logger)
#     config.merge(kwargs)
#     run_episodes(EnsembleDDPGAgent(config))

def plan_ensemble_ddpg(game, log_dir=None, **kwargs):
    config = Config()
    kwargs.setdefault('tag', plan_ensemble_ddpg.__name__)
    kwargs.setdefault('critic_loss_weight', 10.0)
    kwargs.setdefault('num_actors', 5)
    kwargs.setdefault('depth', 2)
    kwargs.setdefault('align_next_v', False)
    kwargs.setdefault('detach_action', False)

    if log_dir is None:
        log_dir = get_default_log_dir(kwargs['tag'])
    task_fn = lambda **kwargs: Bullet(game, **kwargs)
    config.task_fn = lambda: ProcessTask(task_fn)
    config.evaluation_env = ProcessTask(task_fn, log_dir=log_dir)
    config.network_fn = lambda state_dim, action_dim: PlanEnsembleDeterministicNet(
        body=FCBody(state_dim, (400, ), gate=F.tanh), action_dim=action_dim, num_actors=kwargs['num_actors'],
        discount=config.discount, detach_action=kwargs['detach_action'])
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.max_steps = 1e6
    config.random_process_fn = lambda action_dim: GaussianProcess(
        action_dim, LinearSchedule(0.3, 0, 1e6))
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = Logger('./log', logger)
    config.merge(kwargs)
    run_episodes(PlanEnsembleDDPGAgent(config))

def multi_runs(game, fn, tag, **kwargs):
    runs = np.arange(0, 5)
    for run in runs:
        log_dir = './log/ensemble-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
        fn(game, log_dir, **kwargs)

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=10, max_timesteps=1e6)
    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x, y = plotter.average(data, 100, 1e6, top_k=kwargs['top_k'])
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i==0 else '')
    plt.legend()
    plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

if __name__ == '__main__':
    mkdir('data')
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    os.system('export OMP_NUM_THREADS=1')
    os.system('export MKL_NUM_THREADS=1')
    torch.set_num_threads(1)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # game = 'RoboschoolAnt-v1'
    game = 'AntBulletEnv-v0'

    # multi_runs(game, ddpg_continuous, tag='original_ddpg_tanh',
    #                 gate=F.tanh, q_l2_weight=0)

    # multi_runs(game, ddpg_continuous, tag='original_ddpg_relu',
    #                 gate=F.relu, q_l2_weight=0)

    # multi_runs(game, ddpg_continuous, tag='original_ddpg_l2_relu',
    #                 gate=F.relu, q_l2_weight=0.01)

    # plan_ensemble_ddpg(game, tag='plan_ensemble_detach',
    #                    depth=2, num_actors=5, detach_action=True)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_no_detach',
    #                    depth=2, num_actors=5, detach_action=False)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_original',
    #                    depth=2, num_actors=5, align_next_v=False, detach_action=False)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_align_next_v',
    #                    depth=2, num_actors=5, align_next_v=True, detach_action=False)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_detach_action',
    #                    depth=2, num_actors=5, align_next_v=False, detach_action=True)
    # plan_ensemble_ddpg(game, tag='plan_ensemble_depth_1',
    #                    depth=1, num_actors=5, align_next_v=False, detach_action=False)

    # plan_ensemble_ddpg(game, depth=2, num_actors=5)
    # d3pg_conginuous(game, num_actors=1)

    # multi_runs(game, ddpg_continuous, tag='original_ddpg')
    # multi_runs(game, ensemble_ddpg, tag='5_actors', num_actors=5)
    # multi_runs(game, plan_ensemble_ddpg, tag='5_actors')

    # ensemble_ddpg(game, num_actors=10, tag='ensemble_ddpg_run_1')

    # plot(pattern='.*plan_ensemble_ddpg.*', figure=0)
    # plt.show()

    # plot(pattern='.*plan_ensemble_align_next_v.*', figure=0)
    # plot(pattern='.*plan_ensemble_depth.*', figure=1)
    # plot(pattern='.*plan_ensemble_detach.*', figure=0)
    # plot(pattern='.*plan_ensemble_no_detach.*', figure=1)
    # plot(pattern='.*plan_ensemble_original.*', figure=3)
    # plot(pattern='.*plan_ensemble_new_impl.*', figure=4)
    # plt.show()

    # top_k = 0
    # plot(pattern='.*ensemble-%s.*ddpg_continuous.*' % (game), figure=0, color=0, top_k=top_k)
    # plot(pattern='.*ensemble-%s.*ensemble_ddpg.*5_actors.*' % (game), figure=0, color=1, top_k=top_k)
    # plt.show()

    # plot(pattern='.*ensemble-%s.*original_ddpg.*' % (game), figure=0)
    # plot(pattern='.*ensemble-%s.*5_actors.*' % (game), figure=1)
    # plot(pattern='.*ensemble-%s.*10_actors.*' % (game), figure=2)
    # plt.show()

    # plot(pattern='.*ensemble_ddpg.*', figure=0)
    # plot(pattern='.*hopper_ensemble_ddpg.*', figure=1)
    # plot(pattern='.*expert-RoboschoolHopper.*', figure=0)
    # plot(pattern='.*expert-RoboschoolReacher.*', figure=0)
    # plt.show()

    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1/ddpg_continuous.*.ddpg_l2_relu.*', figure=0)
    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1/ddpg_continuous.*.ddpg_relu.*', figure=1)
    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1/ddpg_continuous.*.ddpg_tanh.*', figure=2)
    # plt.show()

    plot(pattern='.*ddpg_continuous-180511-095212.*', figure=0)
    plot(pattern='.*ddpg_continuous-180511-095404.*', figure=0)
    plot(pattern='.*plan_ensemble_detach-180511-100739.*', figure=1)
    plot(pattern='.*plan_ensemble_detach-180511-100747.*', figure=1)
    plt.show()


