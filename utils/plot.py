# from https://raw.githubusercontent.com/openai/baselines/master/baselines/results_plotter.py
__all__ = ['plot_results']

import numpy as np
import matplotlib.pyplot as plt
from component import load_results
plt.rcParams['svg.fonttype'] = 'none'

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
        y = ts.r.values
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
        y = ts.r.values
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
        y = ts.r.values
    else:
        raise NotImplementedError
    return x, y

def plot_curves(xy_list, xaxis, title):
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        # plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()

def plot_results(dirs, num_timesteps, xaxis, task_name):
    tslist = []
    for dir in dirs:
        ts = load_results(dir)
        ts = ts[ts.l.cumsum() <= num_timesteps]
        tslist.append(ts)
    xy_list = [ts2xy(ts, xaxis) for ts in tslist]
    plot_curves(xy_list, xaxis, task_name)

if __name__ == '__main__':
    plot_results(['../log/CartPole-v0-vanilla'], 10e6, X_TIMESTEPS, "CartPole")
    plt.show()
