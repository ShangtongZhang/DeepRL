import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_rl import *

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    # kwargs.setdefault('top_k_perf', lambda x: np.mean(x[-20:]))
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('episode_window', 100)
    kwargs.setdefault('x_interval', 1000)
    kwargs.setdefault('down_sample', True)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=kwargs['episode_window'], max_timesteps=kwargs['max_timesteps'])
    print('')

    figure = kwargs['figure']
    plt.figure(figure)
    if kwargs['average']:
        color = kwargs['color']
        x, y = plotter.average(data, kwargs['x_interval'], kwargs['max_timesteps'], top_k=kwargs['top_k'])
        print(y.shape)
        if kwargs['down_sample']:
            indices = np.linspace(0, len(x) - 1, 500).astype(np.int)
            x = x[indices]
            y = y[:, indices]
        name = names[0].split('/')[-1]
        plotter.plot_standard_error(y, x, label=name, color=Plotter.COLORS[color])
        # sns.tsplot(y, x, condition=name, , ci='sd')
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
    if 'y_lim' in kwargs.keys():
        plt.ylim(kwargs['y_lim'])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')

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
        data = [plotter.window_func(x, y, kwargs['episode_window'], np.mean) for x, y in data]
        x = data[0][0]
        y = [y for x, y in data]
        y = np.stack(y)
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color], ci='sd')
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            if 'color' not in kwargs.keys():
                color = Plotter.COLORS[i]
            else:
                color = Plotter.COLORS[kwargs['color']]
            plt.plot(x, y, color=color, label=name if i==0 else '')
    plt.legend()
    if 'y_lim' in kwargs:
        plt.ylim(kwargs['y_lim'])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

def plot_improvement():
    games = [
        'Freeway',
        'BeamRider',
        'BattleZone',
        'Robotank',
        'Qbert',
        'Alien',
        'Amidar',
        'Seaquest',
        'MsPacman',
        'Enduro',
        'Assault',
        'Asterix',
        'Asteroids',
        'Atlantis',
        'BankHeist',
        'Bowling',
        'Boxing',
        'Breakout',
        'Centipede',
        'ChopperCommand',
        'CrazyClimber',
        'DemonAttack',
        'DoubleDunk',
        'FishingDerby',
        'Frostbite',
        'Gopher',
        'Gravitar',
        'IceHockey',
        'Jamesbond',
        'Kangaroo',
        'Krull',
        'KungFuMaster',
        'MontezumaRevenge',
        'NameThisGame',
        'Pitfall',
        'Pong',
        'PrivateEye',
        'Riverraid',
        'RoadRunner',
        'SpaceInvaders',
        'StarGunner',
        'Tennis',
        'TimePilot',
        'Tutankham',
        'UpNDown',
        'Venture',
        'VideoPinball',
        'WizardOfWor',
        'Zaxxon'
    ]

    patterns = [
        't001b001_s_le',
        'n_step_qr_dqn',
        'n_step_qr_le_dqn',
    ]

    cum_rewards = {}

    plotter = Plotter()
    for game in games:
        for p in patterns:
            names = plotter.load_log_dirs(pattern='.*dist_rl.*%s.*%s.*train.*' % (game, p))
            if len(names) == 0:
                print('data not found: %s, %s' % (game, p))
                continue
            data = plotter.load_results(names, episode_window=0, max_timesteps=int(4e7))
            cum_y = [np.sum(y) for x, y in data]
            final_y = [np.sum(y[-1000: ]) for x, y in data]
            if game not in cum_rewards.keys():
                cum_rewards[game] = []
            cum_rewards[game].extend([np.mean(cum_y), np.mean(final_y)])

    improvements = [[], [], [], []]
    for game in cum_rewards.keys():
        stats = cum_rewards[game]
        print(game, len(stats))
        new_cum = stats[0]
        new_final = stats[1]
        base1_cum = stats[2]
        base1_final = stats[3]
        base2_cum = stats[4]
        base2_final = stats[5]
        improvements[0].append([game, (new_cum - base1_cum) / base1_cum])
        improvements[1].append([game, (new_final - base1_final) / base1_final])
        improvements[2].append([game, (new_cum - base2_cum) / base2_cum])
        improvements[3].append([game, (new_final - base2_final) / base2_final])
    data = [zip(*sorted(ratio, key=lambda x: x[1])) for ratio in improvements]
    import matplotlib.pyplot as plt
    for i, (x, y) in enumerate(data):
        plt.figure(i)
        plt.figure(figsize=(30, 4))
        plt.tight_layout()
        bar = plt.bar(x, y)
        plt.xticks(rotation='vertical')
        # plt.ylabel('improvement')
        plt.subplots_adjust(bottom=0.25)

        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
        for j, bari in enumerate(bar):
            if y[j] >= 0:
                h = bari.get_height()
                va = 'bottom'
            else:
                h = 0
                # va = 'top'
                va = 'bottom'
            v = '%.1f%%' % (y[j] * 100)
            plt.text(bari.get_x() + bari.get_width() / 2, h, v, va=va,
                           ha='center', color='black', fontsize=15, rotation='vertical')
        plt.savefig('/Users/Shangtong/Dropbox/Paper/quantile_option/img/atari-%d.png' % (i), bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    # plot_improvement()
    games = [
        'Freeway',
        'BeamRider',
        'BattleZone',
        'Robotank',
        'Qbert',
        'Alien',
        'Amidar',
        'Seaquest',
        'MsPacman',
        'Enduro',
        'Assault',
        'Asterix',
        'Asteroids',
        'Atlantis',
        'BankHeist',
        'Bowling',
        'Boxing',
        'Breakout',
        'Centipede',
        'ChopperCommand',
        'CrazyClimber',
        'DemonAttack',
        'DoubleDunk',
        'FishingDerby',
        'Frostbite',
        'Gopher',
        'Gravitar',
        'IceHockey',
        'Jamesbond',
        'Kangaroo',
        'Krull',
        'KungFuMaster',
        'MontezumaRevenge',
        'NameThisGame',
        'Pitfall',
        'Pong',
        'PrivateEye',
        'Riverraid',
        'RoadRunner',
        'SpaceInvaders',
        'StarGunner',
        'Tennis',
        'TimePilot',
        'Tutankham',
        'UpNDown',
        'Venture',
        'VideoPinball',
        'WizardOfWor',
        'Zaxxon'
    ]

    # games = [
    #     'BreakoutNoFrameskip-v4',
    #     'AssaultNoFrameskip-v4',
    #     'JamesbondNoFrameskip-v4',
    #     'DemonAttackNoFrameskip-v4'
    # ]

    train_kwargs = {
        'episode_window': 100,
        'top_k': 0,
        'max_timesteps': int(4e7),
        # 'max_timesteps': int(3e7),
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
        'original',
        't0b0_ns',
        't01b01_ns',
        't001b001_ns',
        # 't0b0_s',
    ]

    patterns = [
        't001b001_s_le',
        # 't001b001_s_se',
        'n_step_qr_dqn',
        'n_step_qr_le_dqn',
        # 'n_step_dqn',
        # 'n_step_dqn',
    ]

    for j, game in enumerate(games):
        for i, p in enumerate(patterns):
            try:
                plot(pattern='.*dist_rl.*%s.*%s.*train.*' % (game, p), figure=j, color=i, **train_kwargs)
                plt.savefig('data/dist_rl_images/n-step-qr-dqn-%s-train.png' % (game))
            except Exception as e:
                print(e)
                continue
            # plot(pattern='.*dist_rl.*%s.*%s.*test.*' % (game, p), figure=j, color=i, **train_kwargs)
            # plt.savefig('data/dist_rl_images/n-step-qr-dqn-%s-test.png' % (game))
            # deterministic_plot(pattern='.*dist-rl.*%s.*%s.*test.*' % (game, p), figure=j, color=i, **test_kwargs)
            # plt.savefig('data/dist_rl_images/n-step-qr-dqn-%s-test.png' % (game))
    # plt.show()

    train_kwargs = {
        'episode_window': 5000,
        'top_k': 0,
        'max_timesteps': int(4e7),
        # 'max_timesteps': int(2e7),
        'average': True,
        'x_interval': 1000,
        # 'y_lim': [-2, 5],
        'down_sample': True,
    }
    test_kwargs = {
        'episode_window': 50,
        'average': True,
        'x_interval': 16e4,
        'rep': 10,
        'max_timesteps': int(4e7),
        'y_lim': [-2, 5]
    }

    tag = 'le'
    patterns = [
        't0b0_s_le',
        't001b001_s_le',
        't01b01_s_le',
        'original',
    ]

    # for i, p in enumerate(patterns):
    #     plot(pattern='.*dist_rl-IceCliff.*%s.*-train.*' %(p), figure=0, color=i, **train_kwargs)
    #     plt.savefig('data/dist_rl_images/n-step-qr-dqn-IceCliff-%s-train.png' % (tag))
    #     plot(pattern='.*dist_rl-IceCliff.*%s.*-test.*' %(p), figure=1, color=i, **train_kwargs)
    #     plt.savefig('data/dist_rl_images/n-step-qr-dqn-IceCliff-%s-test.png' % (tag))
    #     # deterministic_plot(pattern='.*dist_rl-IceCliff.*%s.*-test.*' %(p), figure=1, color=i, **test_kwargs)
    #     # plt.savefig('data/dist_rl_images/n-step-qr-dqn-IceCliff-%s-test.png' % (train_kwargs['tag']))
    # plt.show()

    train_kwargs = {
        'average': True,
        'x_interval': 1000,
        'top_k': 0,
        'max_timesteps': int(3e5),
        'down_sample': False
    }
    test_kwargs = {
        'average': True,
        'x_interval': 1600,
        'rep': 20,
        'max_timesteps': int(3e6),
    }
    patterns = [
        'original',
        't0b0ns',
        't0b0s',
    ]
    # for i, p in enumerate(patterns):
    #     plot(pattern='.*dist_rl-CliffWalking.*bootstrapped_qr_dqn_cliff.*%s.*train.*' % (p), figure=0, color=i, **train_kwargs)
    # plt.show()

    patterns = [
        'original',
        'b001_se',
        'b001_me',
        'b001_le',
        'b0_se',
        'b0_me',
        'b0_le',
        'b1_se',
        'b1_me',
        'b1_le'
    ]
    # for i, p in enumerate(patterns):
    #     plot(pattern='.*replay_qo.*%s.*' % (p), figure=0, color=i)
    # plt.show()



