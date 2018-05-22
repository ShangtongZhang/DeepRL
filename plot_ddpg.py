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
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i==0 else '')
    plt.legend()
    # plt.ylim([-200, 1400])
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
    # plt.show()

    # games = ['Walker2DBulletEnv-v0',
    #          'AntBulletEnv-v0',
    #          'HopperBulletEnv-v0',
    #          'RacecarBulletEnv-v0',
    #          'KukaBulletEnv-v0',
    #          'MinitaurBulletEnv-v0']
    # game = games[0]
    games = ['RoboschoolAnt-v1',
             'RoboschoolHalfCheetah-v1',
             'RoboschoolHopper-v1',
             'RoboschoolInvertedDoublePendulum-v1',
             'RoboschoolReacher-v1',
             'RoboschoolWalker2d-v1']
    # games = games[:1]

    # game = 'ensemble-RoboschoolAnt-v1'
    # game = 'ensemble-RoboschoolHalfCheetah-v1'
    # game = 'ensemble-RoboschoolHopper-v1'
    # game = 'ensemble-RoboschoolWalker2d-v1'
    # game = 'RoboschoolHopper-v1'
    # game = 'RoboschoolHumanoid-v1'
    # plot(pattern='.*%s.*ddpg_continuous.*' % (game), figure=0, average=False)
    # plot(pattern='.*%s.*plan_ensemble_detach.*' % (game), figure=1)
    # plot(pattern='.*%s.*plan_ensemble_no_detach.*' % (game), figure=2)
    # plt.show()

    # plot(pattern='.*d3pg_conginuous-180514-114241.*', figure=0)
    # plot(pattern='.*%s.*original_d3pg.*' % (game), figure=0, average=True, max_timesteps=5e6, color=0)
    # plot(pattern='.*%s.*5_actors.*' % (game), figure=0, average=True, max_timesteps=5e6, color=1)
    # plt.show()

    # plot(pattern='.*%s.*original_ddpg.*' % (game), figure=0)
    # plot(pattern='.*%s.*ensemble_off_policy.*' % (game), figure=1)
    # plot(pattern='.*%s.*ensemble_on_policy.*' % (game), figure=2)

    # plot(pattern='.*%s.*original_ddpg.*' % (game), figure=0, average=True, max_timesteps=1e6, color=0)
    # plot(pattern='.*%s.*ensemble_off_policy.*' % (game), figure=0, average=True, max_timesteps=1e6, color=1)
    # plot(pattern='.*%s.*ensemble_on_policy.*' % (game), figure=0, average=True, max_timesteps=1e6, color=2)
    # plot(pattern='.*%s.*ensemble_half_off_policy.*' % (game), figure=0, average=True, max_timesteps=1e6, color=3)
    # plt.show()

    # plot(pattern='.*d3pg_ensemble-180516-120151.*', figure=0)
    # plot(pattern='.*d3pg_ensemble-180516-120159.*', figure=1)
    # plot(pattern='.*d3pg_ensemble-180516-121318.*', figure=2)
    # plt.show()

    # plot(pattern='.*log/option_no_beta_d3pg/%s/d3pg_conginuous/original_d3pg.*' % (game), figure=0, average=True, color=0, max_timesteps=1e7)
    # plot(pattern='.*log/option_no_beta_d3pg/%s/d3pg_ensemble/half_policy.*' % (game), figure=0, average=True, color=1, max_timesteps=1e7)
    # plot(pattern='.*log/option_no_beta_d3pg/%s/d3pg_ensemble/on_policy.*' % (game), figure=0, average=True, color=2, max_timesteps=1e7)
    # plot(pattern='.*log/option_no_beta_d3pg/%s/d3pg_ensemble/off_policy.*' % (game), figure=0, average=True, color=3, max_timesteps=1e7)
    # plt.show()

    # plot(pattern='.*a2c_pixel_atari-180518-102724.*', figure=0)
    # plot(pattern='.*ppo_pixel_atari-180518-102743.*', figure=1)
    # plot(pattern='.*ddpg_pixel.*', figure=2)
    # plt.show()

    # for i, game in enumerate(games):
    #     plot(pattern='.*option_no_beta_d3pg.*ensemble-%s.*original_d3pg.*' % (game), figure=i, average=True, max_timesteps=1e7, color=0)
    #     plot(pattern='.*option_no_beta_d3pg.*ensemble-%s.*half_policy.*' % (game), figure=i, average=True, max_timesteps=1e7, color=1)
    #     plot(pattern='.*option_no_beta_d3pg.*ensemble-%s.*on_policy.*' % (game), figure=i, average=True, max_timesteps=1e7, color=2)
    #     plot(pattern='.*option_no_beta_d3pg.*ensemble-%s.*off_policy.*' % (game), figure=i, average=True, max_timesteps=1e7, color=3)
    # plt.show()

    games = [
        'ensemble-AntBulletEnv-v0',
        'ensemble-HalfCheetahBulletEnv-v0',
        'ensemble-HopperBulletEnv-v0',
        'ensemble-Walker2DBulletEnv-v0'
    ]
    for i, game in enumerate(games):
        plot(pattern='.*log/%s.*' % (game), figure=i)
    plt.show()
