# Adapted from  https://github.com/openai/baselines/blob/master/baselines/results_plotter.py

import numpy as np
import component

class Plotter:
    COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
              'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
              'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

    X_TIMESTEPS = 'timesteps'
    X_EPISODES = 'episodes'
    X_WALLTIME = 'walltime_hrs'

    def __init__(self):
        pass

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def window_func(self, x, y, window, func):
        yw = self.rolling_window(y, window)
        yw_func = func(yw, axis=-1)
        return x[window - 1:], yw_func

    def ts2xy(self, ts, xaxis):
        if xaxis == Plotter.X_TIMESTEPS:
            x = np.cumsum(ts.l.values)
            y = ts.r.values
        elif xaxis == Plotter.X_EPISODES:
            x = np.arange(len(ts))
            y = ts.r.values
        elif xaxis == Plotter.X_WALLTIME:
            x = ts.t.values / 3600.
            y = ts.r.values
        else:
            raise NotImplementedError
        return x, y

    def load_results(self, dirs, max_timesteps=1e8, x_axis=X_TIMESTEPS, episode_window=100):
        tslist = []
        for dir in dirs:
            ts = component.load_monitor_log(dir)
            ts = ts[ts.l.cumsum() <= max_timesteps]
            tslist.append(ts)
        xy_list = [self.ts2xy(ts, x_axis) for ts in tslist]
        if episode_window:
            xy_list = [self.window_func(x, y, episode_window, np.mean) for x, y in xy_list]
        return xy_list

    def plot_results(self, dirs, max_timesteps=1e8, x_axis=X_TIMESTEPS, episode_window=100, title=None):
        import matplotlib.pyplot as plt
        plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 1))
        xy_list = self.load_results(dirs, max_timesteps, x_axis, episode_window)
        for (i, (x, y)) in enumerate(xy_list):
            color = Plotter.COLORS[i]
            plt.plot(x, y, color=color)
        plt.xlabel(x_axis)
        plt.ylabel("Episode Rewards")
        if title is not None:
            plt.title(title)
