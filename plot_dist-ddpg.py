import matplotlib.pyplot as plt
from deep_rl import *

def compute_stats(**kwargs):
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=1e8)
    max_rewards = []
    for x, y in data:
        max_rewards.append(np.max(y))
    return np.mean(max_rewards), np.std(max_rewards) / np.sqrt(len(max_rewards))

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=50, max_timesteps=kwargs['max_timesteps'])
    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x, y = plotter.average(data, 100, kwargs['max_timesteps'], top_k=kwargs['top_k'])
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color], ci='sd')
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i==0 else '')
    plt.legend()
    # plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

def ddpg_plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=kwargs['max_timesteps'])
    data = [y[: len(y) // kwargs['rep'] * kwargs['rep']] for x, y in data]
    min_y = np.min([len(y) for y in data])
    data = [y[ :min_y] for y in data]
    new_data = []
    for y in data:
        y = np.reshape(np.asarray(y), (-1, kwargs['rep'])).mean(-1)
        x = np.arange(y.shape[0]) * kwargs['x_interval']
        new_data.append([x, y])
    data = new_data

    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x = data[0][0]
        y = [entry[1] for entry in data]
        # y = np.transpose(np.stack(y))
        y = np.stack(y)
        name = names[0].split('/')[-1]
        sns.tsplot(y, x, condition=name, color=Plotter.COLORS[color], ci='sd')
        plt.title(names[0])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i==0 else '')
    plt.legend()
    # plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

def plot_sub(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=kwargs['max_timesteps'])
    data = [y[: len(y) // kwargs['rep'] * kwargs['rep']] for x, y in data]
    min_y = np.min([len(y) for y in data])
    data = [y[ :min_y] for y in data]
    new_data = []
    for y in data:
        y = np.reshape(np.asarray(y), (-1, kwargs['rep'])).mean(-1)
        x = np.arange(y.shape[0]) * kwargs['x_interval']
        new_data.append([x, y])
    data = new_data

    print('')

    color = kwargs['color']
    x = data[0][0]
    y = [entry[1] for entry in data]
    y = np.stack(y)
    plotter.plot_standard_error(y, x, label=kwargs['label'], color=Plotter.COLORS[color])
    plt.title(kwargs['name'], fontsize=30)
    plt.xticks([0, int(1e6)], ['0', '1M'])

    # plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    # plt.xlabel('timesteps')
    # plt.ylabel('episode return')
    # plt.show()

def plot_half():
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True
    }
    games = [
        'Ant',
        'Walker2d',
        'Hopper',
        'HalfCheetah',
        'Reacher',
        'Pong',
        # 'Humanoid',
        # 'HumanoidFlagrun',
        # 'HumanoidFlagrunHarder',
        # 'InvertedPendulum',
        # 'InvertedPendulumSwingup',
        # 'InvertedDoublePendulum',
    ]

    patterns = [
        'b1e0',
    ]

    plt.figure(figsize=(30, 4))
    for j, game in enumerate(games):
        plt.subplot(1, 6, j+1)
        for i, p in enumerate(patterns):
            plot_sub(pattern='.*log/option-ddpg/option-Roboschool%s-v1.*%s.*' % (game, p),
                     figure=j, color=i, name=game, label='QUOTA', **kwargs)
        plot_sub(pattern='.*log/baseline-ddpg/baseline-Roboschool%s-v1/ddpg_continuous.*' % (game),
                 figure=j, color=i+1, name=game, label='DDPG', **kwargs)
        plot_sub(pattern='.*log/baseline-ddpg/baseline-Roboschool%s-v1/.*q_ddpg.*' % (game),
                 figure=j, color=i+2, name=game, label='QR-DDPG', **kwargs)
    plt.subplot(1, 6, 1)
    plt.legend()
    plt.savefig('/Users/Shangtong/Dropbox/Paper/quantile_option/aaai19_crc/roboschool.png', bbox_inches='tight')


def plot_all():
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True
    }
    games = [
        'Ant',
        'Walker2d',
        'Hopper',
        'HalfCheetah',
        'Reacher',
        'Pong',
        'Humanoid',
        'HumanoidFlagrun',
        'HumanoidFlagrunHarder',
        'InvertedPendulum',
        'InvertedPendulumSwingup',
        'InvertedDoublePendulum',
    ]

    patterns = [
        'b1e0',
    ]

    plt.figure(figsize=(40, 30))
    for j, game in enumerate(games):
        plt.subplot(3, 4, j + 1)
        for i, p in enumerate(patterns):
            plot_sub(pattern='.*log/option-ddpg/option-Roboschool%s-v1.*%s.*' % (game, p), figure=j, color=i, name=game,
                     **kwargs)
        plot_sub(pattern='.*log/baseline-ddpg/baseline-Roboschool%s-v1/ddpg_continuous.*' % (game), figure=j,
                 color=i+1, name=game, **kwargs)
        plot_sub(pattern='.*log/baseline-ddpg/baseline-Roboschool%s-v1/.*q_ddpg.*' % (game), figure=j, color=i+2,
                 name=game, **kwargs)
    plt.savefig('/Users/Shangtong/Dropbox/Paper/quantile_option/img/roboschool_all.png', bbox_inches='tight')

if __name__ == '__main__':
    plot_half()
    # plot_all()
