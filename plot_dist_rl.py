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
        x, y = plotter.average(data, 1000, kwargs['max_timesteps'], top_k=kwargs['top_k'])
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            if len(names) > 1:
                color = Plotter.COLORS[i]
            else:
                color = Plotter.COLORS[kwargs['color']]
            plt.plot(x, y, color=color, label=name if i==0 else '')
    plt.legend()
    # plt.ylim([0, 400])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

if __name__ == '__main__':
    # plot(pattern='.*n_step_dqn_pixel_atari-180517-092901.*', figure=0)
    # plot(pattern='.*n_step_qr_dqn_pixel_atari-180517-092933.*', figure=1)
    # plot(pattern='.*quantile_regression_dqn_pixel_atari-180517-121816.*', figure=2)
    # plt.show()

    # plot(pattern='.*5_options.*', figure=0)
    # plot(pattern='.*10_options.*', figure=1)
    # plot(pattern='.*20_options.*', figure=2)
    # plot(pattern='.*option_qr_10_options-180519-220446.*', figure=0, color=0)
    # plot(pattern='.*qr_dqn-180519-220432.*', figure=0, color=1)
    # plot(pattern='.*ddpg_pixel-180519-220945.*', figure=0)
    # plot(pattern='.*ddpg_pixel-180519-221016.*', figure=1)
    # plt.show()
    # plot(pattern='.*log/dist_rl-FreewayNoFrameskip-v4/option_qr_dqn_pixel_atari.*', figure=0)
    # plot(pattern='.*log/dist_rl-FreewayNoFrameskip-v4/qr_dqn_pixel_atari.*', figure=1)
    # plt.show()

    plot(pattern='.*log/dist_rl-FreewayNoFrameskip-v4/option_qr_dqn_pixel_atari.*', figure=0, average=True, color=0, max_timesteps=3e7)
    plot(pattern='.*log/dist_rl-FreewayNoFrameskip-v4/qr_dqn_pixel_atari.*', figure=0, average=True, color=1, max_timesteps=3e7)
    plt.show()

