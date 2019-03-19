#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import os
import re


class Plotter:
    COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
              'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
              'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

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
            plt.xlabel('steps')
            if not i:
                plt.ylabel(kwargs['tag'])
            plt.title(game)
            plt.legend()
