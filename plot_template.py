import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_rl import *

def plot(**kwargs):
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    # kwargs.setdefault('top_k_perf', lambda x: np.mean(x[-20:]))
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('episode_window', 100)
    kwargs.setdefault('x_interval', 1000)
    kwargs.setdefault('down_sample', False)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=kwargs['episode_window'], max_timesteps=kwargs['max_timesteps'])
    print('')

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

def plot_atari():
    train_kwargs = {
        'episode_window': 100,
        'top_k': 0,
        'max_timesteps': int(2e7),
        # 'max_timesteps': int(3e7),
        'average': False,
        'x_interval': 100
    }

    games = ['Breakout', 'Alien']

    patterns = [
        # 'mix_nq_aux_r5',
        'mix_nq_rmix_r5',
        'mix_nq_study_r5'
    ]

    l = len(games)
    plt.figure(figsize=(l * 10, 10))
    for j, game in enumerate(games):
        plt.subplot(1, l, j + 1)
        for i, p in enumerate(patterns):
            plot(pattern='.*rmix/.*%s.*%s.*' % (game, p), **train_kwargs, figure=j, color=i)
    plt.show()

def ddpg_plot(**kwargs):
    kwargs.setdefault('average', True)
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

    game = kwargs['name']
    color = kwargs['color']
    if kwargs['average']:
        x = data[0][0]
        y = [entry[1] for entry in data]
        y = np.stack(y)
        name = names[0].split('/')[-1]
        plotter.plot_standard_error(y, x, label=name, color=Plotter.COLORS[color])
        plt.title(game)
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[color], label=name if i==0 else '')
    plt.legend()
    # plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')

def plot_mujoco():
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True
    }
    games = [
        'Walker2d-v2',
        'Hopper-v2',
        'HalfCheetah-v2',
        # 'Reacher-v2',
        'Swimmer-v2',
    ]

    patterns = [
        'remark_ddpg-run',
        'remark_ucb-run',
    ]

    l = len(games)
    plt.figure(figsize=(l * 10, 10))
    for j, game in enumerate(games):
        plt.subplot(1, l, j+1)
        for i, p in enumerate(patterns):
            ddpg_plot(pattern='.*exp-ddpg/%s-%s.*' % (game, p), color=i, name=game, **kwargs)
    plt.show()

if __name__ == '__main__':
    plot_mujoco()