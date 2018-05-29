import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from deep_rl import *

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('top_k_perf', lambda x: np.mean(x[-20:]))
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('episode_window', 100)
    kwargs.setdefault('x_interval', 1000)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=kwargs['episode_window'], max_timesteps=kwargs['max_timesteps'])
    print('')

    figure = kwargs['figure']
    plt.figure(figure)
    if kwargs['average']:
        color = kwargs['color']
        x, y = plotter.average(data, kwargs['x_interval'], kwargs['max_timesteps'], top_k=kwargs['top_k'],
                               top_k_perf=kwargs['top_k_perf'])
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            if 'color' not in kwargs.keys():
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

    game = 'Freeway'
    # game = 'Seaquest'
    # game = 'MsPacman'
    # game = 'Frostbite'
    # game = 'Enduro'
    # game = 'JourneyEscape'
    # game = 'Tennis'
    # game = 'Pong'
    # game = 'Boxing'
    # game = 'IceHockey'
    # game = 'Skiing'
    # game = 'SpaceInvaders'
    # game = 'UpNDown'
    # game = 'BeamRider'
    # game = 'Robotank'
    # game = 'BankHeist'
    # game = 'BattleZone'
    # kwargs = {
    #     'average': True
    # }
    # plot(pattern='.*dist_rl_quantile_option_no_skip.*%sNoFrameskip-v4.*9_options_only.*' % (game), figure=0, color=0, max_timesteps=3e7, **kwargs)
    # plot(pattern='.*dist_rl_quantile_option_no_skip.*%sNoFrameskip-v4.*original_qr_dqn.*' % (game), figure=0, color=1, max_timesteps=3e7, **kwargs)
    # plot(pattern='.*dist_rl_quantile_option_no_skip.*%sNoFrameskip-v4.*mean_and_9_options.*' % (game), figure=0, color=2, max_timesteps=3e7, **kwargs)
    # plt.show()


    # plot(pattern='.*log/dist_rl_deterministic/dist_rl-%sNoFrameskip-v4/option_qr_dqn_pixel_atari.*' % (game), figure=0, color=0)
    # plot(pattern='.*log/dist_rl_deterministic/dist_rl-%sNoFrameskip-v4/qr_dqn_pixel_atari' % (game), figure=0, color=1)
    # plt.show()

    # plot(pattern='.*dist_rl_deterministic.*%sNoFrameskip-v4.*original_qr_dqn.*' % (game), figure=0, average=False, color=1, max_timesteps=3e7)
    # plot(pattern='.*dist_rl_deterministic.*%sNoFrameskip-v4.*mean_and_9_options.*' % (game), figure=0, average=False, color=2, max_timesteps=3e7)
    # plt.show()

    # plot(pattern='.*dist_rl_quantile_option_random_skip.*%s.*mean_and_9_options.*' % (game), figure=0, average=True, color=0, max_timesteps=1e7)
    # plot(pattern='.*dist_rl_quantile_option_random_skip.*%s.*qr_dqn_random_skip.*' % (game), figure=0, average=True, color=1, max_timesteps=1e7)
    # plt.show()

    # plot(pattern='.*log/original_qr_dqn-180523-162532.*', figure=0)
    # plot(pattern='.*log/mean_and_9_options-180523-162542.*', figure=1)
    # plt.show()

    # plot(pattern='.*log/PongNoFrameskip-v4-option-qr-180524-170331.*', figure=0)
    # plt.show()

    kwargs = {
        'episode_window': 100,
        'top_k': 10,
        'max_timesteps': int(3e5),
        'average': True,
        'x_interval': 100
    }
    plot(pattern='.*log/dist_rl-CliffWalking/qr_dqn_cliff.*', figure=0, color=0, **kwargs)
    plot(pattern='.*log/dist_rl-CliffWalking/option_qr_dqn_cliff/mean_option_qr_dqn.*', figure=0, color=1, **kwargs)
    plot(pattern='.*log/dist_rl-CliffWalking/option_qr_dqn_cliff/pure_quantiles_option_qr_dqn.*', figure=0, color=2, **kwargs)
    # plot(pattern='.*log/dist_rl-CliffWalking/option_qr_dqn_cliff/random_option_qr_dqn.*', figure=0, color=3, **kwargs)
    plt.show()

