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

if __name__ == '__main__':
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