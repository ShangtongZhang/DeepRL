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
             'JourneyEscapeNoFrameskip-v4',
             ]
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
        'max_timesteps': int(2e7),
        'average': True,
        'x_interval': 1000
    }
    test_kwargs = {
        'average': False,
        'x_interval': 16e4,
        'rep': 10,
        'max_timesteps': int(4e7),
    }
    patterns = [
        'original_qr_dqn',
        'per_episode_random_off_termination',
        'per_episode_decay_off_termination',
        'per_episode_decay_intro_q',
        # '9_options_only',
        # 'mean_and_9_options',
    ]
    # for j, game in enumerate(games):
    #     for i, p in enumerate(patterns):
    #         plot(pattern='.*dist-rl.*%s.*%s.*train.*' % (game, p), figure=j, color=i, **train_kwargs)
    #     # plot(pattern='.*log/dist_rl-%sNoFrameskip-v4.*%s.*train.*' % (game, p), figure=0, color=i, **train_kwargs)
    #     # deterministic_plot(pattern='.*log/dist_rl-%sNoFrameskip-v4.*%s.*test.*' % (game, p), figure=0, color=i, **test_kwargs)
    #     plt.savefig('data/dist_rl_images/n-step-qr-dqn-%s.png' % (game))
    # plt.show()

    train_kwargs = {
        'average': True,
        'x_interval': 1000,
        'top_k': 0,
        'max_timesteps': int(3e5),
    }
    test_kwargs = {
        'average': True,
        'x_interval': 1600,
        'rep': 20,
        'max_timesteps': int(3e6),
    }
    patterns = [
        'original_qr_dqn',
        # 'per_episode_random_off_termination',
        # 'per_episode_decay_off_termination',
        # 'per_episode_decay_intro_q',
        '/1_and_1',
        '/0_and_1',
        '/0_and_0',
        '/1_and_0',
        '/0\.9_and_0\.9',
        '/0\.5_and_0\.5',
        '/0\.1_and_0\.1',
    ]
    for i, p in enumerate(patterns):
        # plot(pattern='.*bootstrapped_qr_dqn_cliff.*%s.*train.*' % (p), figure=0, color=i, **train_kwargs)
        deterministic_plot(pattern='.*dist_rl-CliffWalking/bootstrapped_qr_dqn_cliff.*%s.*test.*' % (p), figure=0, color=i, **test_kwargs)
    plt.show()

