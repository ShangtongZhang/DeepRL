import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from deep_rl import *

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=10, max_timesteps=1e6)
    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x, y = plotter.average(data, 100, 1e6, top_k=kwargs['top_k'])
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i==0 else '')
    plt.legend()
    plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

if __name__ == '__main__':
    # plot(pattern='.*plan_ensemble_ddpg.*', figure=0)
    # plt.show()

    # plot(pattern='.*plan_ensemble_align_next_v.*', figure=0)
    # plot(pattern='.*plan_ensemble_depth.*', figure=1)
    # plot(pattern='.*plan_ensemble_detach.*', figure=0)
    # plot(pattern='.*plan_ensemble_no_detach.*', figure=1)
    # plot(pattern='.*plan_ensemble_original.*', figure=3)
    # plot(pattern='.*plan_ensemble_new_impl.*', figure=4)
    # plt.show()

    # top_k = 0
    # plot(pattern='.*ensemble-%s.*ddpg_continuous.*' % (game), figure=0, color=0, top_k=top_k)
    # plot(pattern='.*ensemble-%s.*ensemble_ddpg.*5_actors.*' % (game), figure=0, color=1, top_k=top_k)
    # plt.show()

    # plot(pattern='.*ensemble-%s.*original_ddpg.*' % (game), figure=0)
    # plot(pattern='.*ensemble-%s.*5_actors.*' % (game), figure=1)
    # plot(pattern='.*ensemble-%s.*10_actors.*' % (game), figure=2)
    # plt.show()

    # plot(pattern='.*ensemble_ddpg.*', figure=0)
    # plot(pattern='.*hopper_ensemble_ddpg.*', figure=1)
    # plot(pattern='.*expert-RoboschoolHopper.*', figure=0)
    # plot(pattern='.*expert-RoboschoolReacher.*', figure=0)
    # plt.show()

    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1/ddpg_continuous.*.ddpg_l2_relu.*', figure=0)
    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1/ddpg_continuous.*.ddpg_relu.*', figure=1)
    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1/ddpg_continuous.*.ddpg_tanh.*', figure=2)
    # plt.show()

    # plot(pattern='.*ddpg_continuous-180511-095212.*', figure=0)
    # plot(pattern='.*ddpg_continuous-180511-095404.*', figure=0)
    # plot(pattern='.*plan_ensemble_detach-180511-100739.*', figure=1)
    # plot(pattern='.*plan_ensemble_detach-180511-100747.*', figure=1)
    plt.show()