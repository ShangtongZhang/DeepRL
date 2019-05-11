import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_rl import *

import numpy as np
import os
import re


class Plotter:
    COLORS = ['blue', 'green', 'red', 'orange', 'purple', 'lime', 'lightblue',
              'lavender',
              'brown',
              'coral', 'cyan', 'magenta', 'black', 'gold',
              'yellow', 'pink',
              'teal', 'coral', 'lightblue', 'lavender', 'turquoise',
              'darkgreen', 'tan', 'salmon', 'lightpurple', 'darkred']

    RETURN_TRAIN = 'episodic_return_train'
    RETURN_TEST = 'episodic_return_test'

    def __init__(self):
        pass

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _window_func(self, x, y, window, func):
        yw = self._rolling_window(y, window)
        yw_func = func(yw, axis=-1)
        return x[window - 1:], yw_func

    def load_results(self, dirs, **kwargs):
        kwargs.setdefault('tag', self.RETURN_TRAIN)
        kwargs.setdefault('right_align', False)
        kwargs.setdefault('window', 0)
        kwargs.setdefault('top_k', 0)
        kwargs.setdefault('top_k_measure', None)
        kwargs.setdefault('interpolation', 100)
        xy_list = self.load_log_dirs(dirs, **kwargs)

        if kwargs['top_k']:
            perf = [kwargs['top_k_measure'](y) for _, y in xy_list]
            top_k_runs = np.argsort(perf)[-kwargs['top_k']:]
            new_xy_list = []
            for r, (x, y) in enumerate(xy_list):
                if r in top_k_runs:
                    new_xy_list.append((x, y))
            xy_list = new_xy_list

        if kwargs['interpolation']:
            x_right = float('inf')
            for x, y in xy_list:
                x_right = min(x_right, x[-1])
            x = np.arange(0, x_right, kwargs['interpolation'])
            y = []
            for x_, y_ in xy_list:
                y.append(np.interp(x, x_, y_))
            y = np.asarray(y)
        else:
            x = xy_list[0][0]
            y = [y for _, y in xy_list]
            x = np.asarray(x)
            y = np.asarray(y)

        return x, y

    def filter_log_dirs(self, pattern, negative_pattern=' ', root='./log', **kwargs):
        dirs = [item[0] for item in os.walk(root)]
        leaf_dirs = []
        for i in range(len(dirs)):
            if i + 1 < len(dirs) and dirs[i + 1].startswith(dirs[i]):
                continue
            leaf_dirs.append(dirs[i])
        names = []
        p = re.compile(pattern)
        np = re.compile(negative_pattern)
        for dir in leaf_dirs:
            if p.match(dir) and not np.match(dir):
                names.append(dir)
                print(dir)
        print('')
        return sorted(names)

    def load_log_dirs(self, dirs, **kwargs):
        xy_list = []
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for dir in dirs:
            event_acc = EventAccumulator(dir)
            event_acc.Reload()
            _, x, y = zip(*event_acc.Scalars(kwargs['tag']))
            xy_list.append([x, y])
        if kwargs['right_align']:
            x_max = float('inf')
            for x, y in xy_list:
                x_max = min(x_max, len(y))
            xy_list = [[x[:x_max], y[:x_max]] for x, y in xy_list]
        if kwargs['window']:
            xy_list = [self._window_func(np.asarray(x), np.asarray(y), kwargs['window'], np.mean) for x, y in xy_list]
        return xy_list

    def plot_mean_standard_error(self, data, x=None, **kwargs):
        import matplotlib.pyplot as plt
        if x is None:
            x = np.arange(data.shape[1])
        e_x = np.std(data, axis=0) / np.sqrt(data.shape[0])
        m_x = np.mean(data, axis=0)
        plt.plot(x, m_x, **kwargs)
        del kwargs['label']
        plt.fill_between(x, m_x + e_x, m_x - e_x, alpha=0.3, **kwargs)

    def plot_median_std(self, data, x=None, **kwargs):
        import matplotlib.pyplot as plt
        if x is None:
            x = np.arange(data.shape[1])
        e_x = np.std(data, axis=0)
        m_x = np.median(data, axis=0)
        plt.plot(x, m_x, **kwargs)
        del kwargs['label']
        plt.fill_between(x, m_x + e_x, m_x - e_x, alpha=0.3, **kwargs)

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        plt.figure(figsize=(l * 5, 5))
        plt.rc('text', usetex=True)
        plt.tight_layout()
        for i, game in enumerate(games):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, p), **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean_standard_error(y, x, label=label, color=color)
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.title(game, fontsize=30, fontweight="bold")
            plt.xticks([0, int(2e6)], ['0', r'$2\times10^6$'])
            plt.tick_params(axis='x', labelsize=30)
            plt.tick_params(axis='y', labelsize=25)
            plt.xlabel('Steps', fontsize=30)
            if not i:
                plt.ylabel('Episode Return', fontsize=30)
                plt.legend(fontsize=10, frameon=False)


FOLDER = '/Users/Shangtong/Dropbox/Paper/actor-actor-critic/img'


def plot_dm(type='mean'):
    plotter = Plotter()
    games = [
        'dm-cartpole-b',
        'dm-reacher',
        'dm-cheetah',
        'dm-fish',
        'dm-walker-1',
        'dm-walker-2',
    ]

    titles = [
        'CartPole',
        'Reacher',
        'Cheetah',
        'Fish',
        'Walker1',
        'Walker2',
    ]

    patterns = [
        'ASC-PPO',
        'AHP',
        'remark_PPOC',
        'remark_PPO-',
        'num_workers_4-remark_IOPG',
        'num_workers_4-remark_OC',
    ]

    labels = [
        'DAC+PPO',
        'AHP+PPO',
        'PPOC',
        'PPO',
        'IOPG',
        'OC',
    ]

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        # plt.figure(figsize=(l * 5, 5))
        plt.figure(figsize=(3 * 5, 2 * 5))
        plt.rc('text', usetex=True)
        plt.tight_layout()
        for i, game in enumerate(games):
            # plt.subplot(1, l, i + 1)
            plt.subplot(2, 3, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, p), **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean_standard_error(y, x, label=label, color=color)
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.title(titles[i], fontsize=30, fontweight="bold")
            if i > 2:
                plt.xlabel('Steps', fontsize=30)
            plt.xticks([0, int(2e6)], ['0', r'$2\times10^6$'])
            plt.tick_params(axis='x', labelsize=30)
            plt.tick_params(axis='y', labelsize=25)
            if not i:
                plt.legend(fontsize=13, frameon=False)
            if i == 0 or i == 3:
                plt.ylabel('Episode Return', fontsize=30)

    plot_games(plotter,
               games=games,
               patterns=patterns,
               agg=type,
               downsample=0,
               labels=labels,
               right_align=False,
               tag=plotter.RETURN_TRAIN,
               root='./log/ASquaredC-dm',
               interpolation=100,
               window=20,
               top_k=0,
               )

    plt.tight_layout()
    plt.savefig('%s/ASquaredC-dm-%s.png' % (FOLDER, type), bbox_inches='tight')
    plt.show()


def plot_mujoco(type='mean'):
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
    ]

    patterns = [
        'ASC-PPO',
        'AHP',
        'remark_PPOC-',
        'remark_PPO-',
        'num_workers_4-remark_IOPG',
        'num_workers_4-remark_OC',
        'num_workers_4-remark_ASC-A2C',
    ]

    labels = [
        'DAC+PPO',
        'AHP+PPO',
        'PPOC',
        'PPO',
        'IOPG',
        'OC',
        'DAC+A2C',
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg=type,
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./log/ASquaredC-mujoco',
                       interpolation=100,
                       window=20,
                       top_k=0,
                       )

    plt.tight_layout()
    plt.savefig('%s/ASquaredC-mujoco-%s.png' % (FOLDER, type), bbox_inches='tight')
    plt.show()


def plot_ablation(type='mean'):
    plotter = Plotter()
    games = [
        'dm-walker-2',
    ]

    patterns = [
        'num_o_2',
        'num_o_4',
        'num_o_8',
    ]

    labels = [
        '2 options',
        '4 options',
        '8 options',
    ]

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        # plt.figure(figsize=(l * 5, 5))
        # plt.figure(figsize=(3 * 5, 2 * 5))
        plt.rc('text', usetex=True)
        plt.tight_layout()
        for i, game in enumerate(games):
            # plt.subplot(1, l, i + 1)
            # plt.subplot(2, 3, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, p), **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean_standard_error(y, x, label=label, color=color)
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.title('DAC-PPO Walker2', fontsize=30, fontweight="bold")
            if i > 2:
                plt.xlabel('Steps', fontsize=30)
            plt.xticks([0, int(2e6)], ['0', r'$2\times10^6$'])
            plt.tick_params(axis='x', labelsize=30)
            plt.tick_params(axis='y', labelsize=25)
            if not i:
                plt.legend(fontsize=13, frameon=False)
            if i == 0 or i == 3:
                plt.ylabel('Episode Return', fontsize=30)

    plot_games(self=plotter,
               games=games,
               patterns=patterns,
               agg=type,
               downsample=0,
               labels=labels,
               right_align=False,
               tag=plotter.RETURN_TRAIN,
               root='./log/ASquaredC-ablation',
               interpolation=100,
               window=20,
               top_k=0,
               )

    plt.tight_layout()
    plt.savefig('%s/ASquaredC-ablation-%s.png' % (FOLDER, type), bbox_inches='tight')
    plt.show()


def plot_ddpg():
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Reacher-v2',
    ]

    patterns = [
        'remark_ddpg',
    ]

    labels = [
        'DDPG'
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TEST,
                       root='./data/benchmark/ddpg',
                       interpolation=0,
                       window=0,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/DDPG.png', bbox_inches='tight')


def plot_atari():
    plotter = Plotter()
    games = [
        'BreakoutNoFrameskip-v4',
    ]

    patterns = [
        'remark_a2c',
        'remark_categorical',
        'remark_dqn',
        'remark_n_step_dqn',
        'remark_option_critic',
        'remark_ppo',
        'remark_quantile',
    ]

    labels = [
        'A2C',
        'C51',
        'DQN',
        'N-Step DQN',
        'OC',
        'PPO',
        'QR-DQN',
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=100,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/benchmark/atari',
                       interpolation=0,
                       window=100,
                       )

    plt.show()
    plt.tight_layout()
    # plt.savefig('images/Breakout.png', bbox_inches='tight')


def plot_misc():
    plotter = Plotter()
    games = [
        # 'BreakoutNoFrameskip-v4',
        'AsterixNoFrameskip-v4',
    ]

    patterns = [
        'beta_reg_0-remark_OC-run',
        'beta_reg_0\.01-remark_OC-run',
    ]

    labels = patterns

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=100,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./tf_log',
                       interpolation=0,
                       window=100,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/Breakout.png', bbox_inches='tight')


def plot_option_occupancy():
    colors = ['red', 'green', 'blue', 'yellow']
    files = [
        'data/ASquaredC/dm-cheetah_episode_999424_options.bin',
        'data/ASquaredC/dm-cheetah_episode_1998848_options.bin',
    ]

    for j, file in enumerate(files):
        plt.figure(figsize=(10, 0.2))
        plt.tight_layout()
        with open(file, 'rb') as f:
            options = pickle.load(f)
        options = np.asarray(options)
        xs = []
        for i in range(4):
            xs.append(np.argwhere(options == i).flatten())

        for i, x in enumerate(xs):
            y = np.ones(len(x))
            plt.scatter(x, y, color=colors[i], marker='|')

        plt.axis('off')
        plt.savefig('%s/options-%d.png' % (FOLDER, j), bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    plot_dm(type='mean')
    # plot_mujoco(type='mean')
    # plot_ablation(type='mean')
    # plot_option_occupancy()
