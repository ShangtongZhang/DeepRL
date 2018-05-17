import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from deep_rl import *

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=100, max_timesteps=kwargs['max_timesteps'])
    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x, y = plotter.average(data, 100, kwargs['max_timesteps'], top_k=kwargs['top_k'])
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i==0 else '')
    plt.legend()
    plt.ylim([0, 400])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

if __name__ == '__main__':
    plot(pattern='.*n_step_dqn_pixel_atari-180517-092901.*', figure=0)
    plot(pattern='.*n_step_qr_dqn_pixel_atari-180517-092933.*', figure=1)
    plt.show()

