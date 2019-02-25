import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_rl import *


FOLDER = '/Users/Shangtong/Dropbox/Paper/true-off-policy-policy-gradient/img'

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
            plt.plot(x, y, color=color, label=name if i == 0 else '')
    plt.legend()
    if 'y_lim' in kwargs.keys():
        plt.ylim(kwargs['y_lim'])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')


def ddpg_plot(**kwargs):
    kwargs.setdefault('average', True)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=kwargs['max_timesteps'])
    data = [y[: len(y) // kwargs['rep'] * kwargs['rep']] for x, y in data]
    min_y = np.min([len(y) for y in data])
    data = [y[:min_y] for y in data]
    new_data = []
    for y in data:
        y = np.reshape(np.asarray(y), (-1, kwargs['rep'])).mean(-1)
        x = np.arange(y.shape[0]) * kwargs['x_interval']
        new_data.append([x, y])
    data = new_data

    if kwargs['top_k']:
        scores = []
        for x, y in data:
            scores.append(np.sum(y))
        best = list(reversed(np.argsort(scores)))
        best = best[:kwargs['top_k']]
        data = [data[i] for i in best]

    print('')

    game = kwargs['game']
    color = kwargs['color']
    if kwargs['average']:
        x = data[0][0]
        y = [entry[1] for entry in data]
        y = np.stack(y)
        kwargs.setdefault('label', names[0].split('/')[-1])
        plotter.plot_standard_error(y, x, label=kwargs['label'], color=Plotter.COLORS[color])
        plt.title(game)
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i == 0 else '')
    plt.xlabel('Timesteps')


def get_pattern(game):
    if game == 'Reacher-v2':
        patterns = [
            'algo_off-pac-eval_interval_10-max_steps_2000-run',
            'algo_ace-eval_interval_10-lam1_0-max_steps_2000-run',
            'algo_geoff-pac-c_coef_0\.01-eval_interval_10-gamma_hat_0\.1-lam1_0-lam2_1-max_steps_2000-run',
        ]
        x_ticks = [[0, int(2e4)], ['0', r'$2\times10^5$']]
    elif game == 'Swimmer-v2':
        patterns = [
            'algo_off-pac-max_steps_500000-run',
            'algo_ace-lam1_0-max_steps_500000-run',
            'algo_geoff-pac-gamma_hat_0\.1-lam1_0-lam2_1-max_steps_500000-run',
        ]
        x_ticks = [[0, int(5e5)], ['0', r'$5\times10^6$']]
    else:
        patterns = [
            'algo_off-pac-run',
            'algo_ace-lam1_0-run',
            'algo_geoff-pac-gamma_hat_0\.1-lam1_0-lam2_1-run',
        ]
        x_ticks = [[0, int(1e5)], ['0', r'$10^6$']]
    return patterns, x_ticks


def plot_mujoco():
    kwargs = {
        'x_interval': int(1e2),
        'rep': 10,
        'average': True,
        'top_k': 0
    }
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Reacher-v2',
        'Swimmer-v2',
    ]

    labels = [
        'Off-PAC',
        'ACE',
        'Geoff-PAC',
    ]

    plt.tight_layout()
    l = len(games)
    plt.figure(figsize=(l * 6, 5))
    for j, game in enumerate(games):
        plt.subplot(1, l, j + 1)
        patterns, x_ticks = get_pattern(game)
        for i, p in enumerate(patterns):
            ddpg_plot(pattern='.*geoff-pac-10/%s.*%s.*' % (game, p), color=i, label=labels[i], game=game, **kwargs)
        plt.xticks(*x_ticks)
        if j == 0:
            plt.ylabel('Episode Return')
            plt.legend()
    plt.savefig('%s/mujoco.png' % (FOLDER), bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    plot_mujoco()
