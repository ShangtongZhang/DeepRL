import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from deep_rl import *

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('top_k_perf', lambda x: np.mean(x[-20:]))
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('episode_window', 100)
    kwargs.setdefault('x_interval', 1000)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=kwargs['episode_window'], max_timesteps=kwargs['max_timesteps'])
    print('')

    figure = kwargs['figure']
    plt.figure(figure)
    if kwargs['average']:
        color = kwargs['color']
        x, y = plotter.average(data, kwargs['x_interval'], kwargs['max_timesteps'], top_k=kwargs['top_k'],
                               top_k_perf=kwargs['top_k_perf'])
        name = names[0].split('/')[-1]
        sns.tsplot(y, x, condition=name, color=Plotter.COLORS[color])
        plt.title(names[0])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            if 'color' not in kwargs.keys():
                color = Plotter.COLORS[i]
            else:
                color = Plotter.COLORS[kwargs['color']]
            plt.plot(x, y, color=color, label=name if i==0 else '')
    plt.legend()
    # plt.ylim([0, 400])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

def deterministic_plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('top_k_perf', lambda x: np.mean(x[-20:]))
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('episode_window', 100)
    kwargs.setdefault('x_interval', 1600)
    kwargs.setdefault('rep', 20)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    raw_data = plotter.load_results(names, episode_window=0, max_timesteps=kwargs['max_timesteps'])
    data = []
    for x, y in raw_data:
        y = y[: len(y) // kwargs['rep'] * kwargs['rep']]
        y = np.reshape(np.asarray(y), (-1, kwargs['rep'])).mean(-1)
        x = np.arange(y.shape[0]) * kwargs['x_interval']
        data.append([x, y])
    print('')

    figure = kwargs['figure']
    plt.figure(figure)
    if kwargs['average']:
        color = kwargs['color']
        x = data[0][0]
        y = [entry[1] for entry in data if len(entry[1]) == 188]
        y = np.stack(y)
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            if 'color' not in kwargs.keys():
                color = Plotter.COLORS[i]
            else:
                color = Plotter.COLORS[kwargs['color']]
            plt.plot(x, y, color=color, label=name if i==0 else '')
    plt.legend()
    # plt.ylim([0, 400])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()


if __name__ == '__main__':
    # plot(pattern='.*n_step_dqn_pixel_atari-180517-092901.*', figure=0)
    # plot(pattern='.*n_step_qr_dqn_pixel_atari-180517-092933.*', figure=1)
    # plot(pattern='.*quantile_regression_dqn_pixel_atari-180517-121816.*', figure=2)
    # plt.show()

    # plot(pattern='.*5_options.*', figure=0)
    # plot(pattern='.*10_options.*', figure=1)
    # plot(pattern='.*20_options.*', figure=2)
    # plot(pattern='.*option_qr_10_options-180519-220446.*', figure=0, color=0)
    # plot(pattern='.*qr_dqn-180519-220432.*', figure=0, color=1)
    # plot(pattern='.*ddpg_pixel-180519-220945.*', figure=0)
    # plot(pattern='.*ddpg_pixel-180519-221016.*', figure=1)
    # plt.show()
    # plot(pattern='.*log/dist_rl-FreewayNoFrameskip-v4/option_qr_dqn_pixel_atari.*', figure=0)
    # plot(pattern='.*log/dist_rl-FreewayNoFrameskip-v4/qr_dqn_pixel_atari.*', figure=1)
    # plt.show()


    # kwargs = {
    #     'average': True
    # }
    # plot(pattern='.*dist_rl_quantile_option_no_skip.*%sNoFrameskip-v4.*9_options_only.*' % (game), figure=0, color=0, max_timesteps=3e7, **kwargs)
    # plot(pattern='.*dist_rl_quantile_option_no_skip.*%sNoFrameskip-v4.*original_qr_dqn.*' % (game), figure=0, color=1, max_timesteps=3e7, **kwargs)
    # plot(pattern='.*dist_rl_quantile_option_no_skip.*%sNoFrameskip-v4.*mean_and_9_options.*' % (game), figure=0, color=2, max_timesteps=3e7, **kwargs)
    # plt.show()


    # plot(pattern='.*log/dist_rl_deterministic/dist_rl-%sNoFrameskip-v4/option_qr_dqn_pixel_atari.*' % (game), figure=0, color=0)
    # plot(pattern='.*log/dist_rl_deterministic/dist_rl-%sNoFrameskip-v4/qr_dqn_pixel_atari' % (game), figure=0, color=1)
    # plt.show()

    # plot(pattern='.*dist_rl_deterministic.*%sNoFrameskip-v4.*original_qr_dqn.*' % (game), figure=0, average=False, color=1, max_timesteps=3e7)
    # plot(pattern='.*dist_rl_deterministic.*%sNoFrameskip-v4.*mean_and_9_options.*' % (game), figure=0, average=False, color=2, max_timesteps=3e7)
    # plt.show()

    # plot(pattern='.*dist_rl_quantile_option_random_skip.*%s.*mean_and_9_options.*' % (game), figure=0, average=True, color=0, max_timesteps=1e7)
    # plot(pattern='.*dist_rl_quantile_option_random_skip.*%s.*qr_dqn_random_skip.*' % (game), figure=0, average=True, color=1, max_timesteps=1e7)
    # plt.show()

    # plot(pattern='.*log/original_qr_dqn-180523-162532.*', figure=0)
    # plot(pattern='.*log/mean_and_9_options-180523-162542.*', figure=1)
    # plt.show()

    # plot(pattern='.*log/PongNoFrameskip-v4-option-qr-180524-170331.*', figure=0)
    # plt.show()

    # kwargs = {
    #     'episode_window': 100,
    #     'top_k': 5,
    #     'max_timesteps': int(3e5),
    #     'average': True,
    #     'x_interval': 100
    # }
    # plot(pattern='.*log/dist_rl-CliffWalking/qr_dqn_cliff/qr_dqn-run.*', figure=0, color=0, **kwargs)
    # plot(pattern='.*log/dist_rl-CliffWalking/option_qr_dqn_cliff/mean_option_qr_dqn.*', figure=0, color=1, **kwargs)
    # plot(pattern='.*log/dist_rl-CliffWalking/option_qr_dqn_cliff/pure_quantiles_option_qr_dqn.*', figure=0, color=2, **kwargs)
    # plot(pattern='.*log/dist_rl-CliffWalking/qr_dqn_cliff/random_option_qr_dqn.*', figure=0, color=3, **kwargs)
    # plt.show()

    # game = 'Freeway'
    # game = 'Seaquest'
    # game = 'MsPacman'
    # game = 'Frostbite'
    # game = 'Enduro'
    # game = 'JourneyEscape'
    # game = 'Tennis'
    # game = 'Pong'
    # game = 'Boxing'
    # game = 'IceHockey'
    # game = 'Skiing'
    # game = 'SpaceInvaders'
    # game = 'UpNDown'
    # game = 'BeamRider'
    # game = 'Robotank'
    # game = 'BankHeist'
    # game = 'BattleZone'
    games = ['FreewayNoFrameskip-v4',
             'BeamRiderNoFrameskip-v4',
             'BattleZoneNoFrameskip-v4',
             'RobotankNoFrameskip-v4',
             'PongNoFrameskip-v4']
    # games = [
    #     'BreakoutNoFrameskip-v4',
    #     'AssaultNoFrameskip-v4',
    #     'JamesbondNoFrameskip-v4',
    #     'QbertNoFrameskip-v4',
    #     'DemonAttackNoFrameskip-v4'
    # ]

    train_kwargs = {
        'episode_window': 100,
        'top_k': 0,
        'max_timesteps': int(1e7),
        'average': True,
        'x_interval': 1000
    }
    test_kwargs = {
        'averate': False,
        'x_interval': 16e4,
        'rep': 10,
        'max_timesteps': int(4e7),
    }
    patterns = [
        'per_episode_qr',
        'per_step_qr',
        'per_episode_decay',
        'per_step_decay',
        'original_qr_dqn',
        # '9_options_only',
        # 'mean_and_9_options',
    ]
    # for j, game in enumerate(games):
    #     for i, p in enumerate(patterns):
    #         plot(pattern='.*dist-rl.*%s.*%s.*' % (game, p), figure=j, color=i, **train_kwargs)
    #     # plot(pattern='.*log/dist_rl-%sNoFrameskip-v4.*%s.*train.*' % (game, p), figure=0, color=i, **train_kwargs)
    #     # deterministic_plot(pattern='.*log/dist_rl-%sNoFrameskip-v4.*%s.*test.*' % (game, p), figure=0, color=i, **test_kwargs)
    #     plt.savefig('data/dist_rl_images/n-step-qr-dqn-%s.png' % (game))
    # plt.show()

    kwargs = {
        'average': True,
        'x_interval': 1000,
        'top_k': 0,
        'max_timesteps': int(3e5),
    }
    # kwargs = {
    #     'average': True,
    #     'x_interval': 100,
    #     'top_k': 0,
    #     'max_timesteps': int(4e4),
    # }
    patterns = [
        'original_qr_dqn-run',
        'per_episode_qr_dqn-run',
        'per_step_qr_dqn-run',
        'per_step_decay',
        'per_episode_decay_qr',
        'per_episode_decay_intro'
    ]
    for i, p in enumerate(patterns):
        # plot(pattern='.*replay_bootstrapped_qr_dqn_cliff.*%s.*' % (p), figure=0, color=i, **kwargs)
        plot(pattern='.*dist_rl-CliffWalking/bootstrapped_qr_dqn_cliff.*%s.*' % (p), figure=0, color=i, **kwargs)
    plt.show()

    # kwargs = {
    #     'episode_window': 100,
    #     'top_k': 0,
    #     'max_timesteps': int(3e6),
    #     'average': False,
    #     'x_interval': 1000
    # }
    # patterns = [
    #     'original_qr_dqn-',
    #     'per_episode-',
    #     'per_step-',
    #     'per_step_decay-',
    #     'per_episode_decay-'
    # ]
    # for i, p in enumerate(patterns):
    #     plot(pattern='.*replay_dist_rl.*%s.*' % (p), figure=0, color=i, **kwargs)
    # plt.show()


