import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from deep_rl import *

def compute_stats(**kwargs):
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=1e8)
    max_rewards = []
    for x, y in data:
        max_rewards.append(np.max(y))
    return np.mean(max_rewards), np.std(max_rewards) / np.sqrt(len(max_rewards))

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=50, max_timesteps=kwargs['max_timesteps'])
    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x, y = plotter.average(data, 100, kwargs['max_timesteps'], top_k=kwargs['top_k'])
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color], ci='sd')
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

def ddpg_plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=kwargs['max_timesteps'])
    data = [y[: len(y) // kwargs['rep'] * kwargs['rep']] for x, y in data]
    min_y = np.min([len(y) for y in data])
    # print(min_y)
    data = [y[ :min_y] for y in data]
    new_data = []
    for y in data:
        y = np.reshape(np.asarray(y), (-1, kwargs['rep'])).mean(-1)
        x = np.arange(y.shape[0]) * kwargs['x_interval']
        new_data.append([x, y])
    data = new_data
    peak = [np.max(y) for x, y in data]
    print(peak)

    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x = data[0][0]
        y = [entry[1] for entry in data]
        # y = np.transpose(np.stack(y))
        y = np.stack(y)
        name = names[0].split('/')[-1]
        sns.tsplot(y, x, condition=name, color=Plotter.COLORS[color], ci='sd')
        plt.title(names[0])
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
    return peak

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

    # games = ['RoboschoolAnt-v1',
    #          'RoboschoolHalfCheetah-v1',
    #          'RoboschoolHopper-v1',
    #          'RoboschoolInvertedDoublePendulum-v1',
    #          'RoboschoolReacher-v1',
    #          'RoboschoolWalker2d-v1',
    #          'RoboschoolInvertedPendulumSwingup-v1']
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

    # games = [
    #     'ensemble-AntBulletEnv-v0',
    #     'ensemble-HalfCheetahBulletEnv-v0',
    #     'ensemble-HopperBulletEnv-v0',
    #     'ensemble-Walker2DBulletEnv-v0'
    # ]
    # for i, game in enumerate(games):
    #     plot(pattern='.*log/%s.*' % (game), figure=i, average=True, max_timesteps=1e6)
    # plt.show()
    # for i, game in enumerate(games):
        # stats = compute_stats(pattern='.*log/ensemble-%s/ddpg_continuous.*' % (game))
        # print(game, 'ddpg_continuous', stats)
        # stats = compute_stats(pattern='.*log/ensemble-%s.*half_policy.*' % (game))
        # print(game, 'half_policy', stats)
        # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-%s.*original_ddpg.*' % (game), figure=i, average=True, max_timesteps=1e6, color=0)
        # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-%s.*half_policy.*' % (game), figure=i, average=True, max_timesteps=1e6, color=1)
        # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-%s.*on_policy.*' % (game), figure=i, average=True, max_timesteps=1e6, color=2)
        # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-%s.*off_policy.*' % (game), figure=i, average=True, max_timesteps=1e6, color=3)
    # plt.show()

    # plot(pattern='.*option_no_beta_exp_replay.*Humanoid.*original_ddpg.*', figure=0, average=True, max_timesteps=1e6, color=0)
    # plot(pattern='.*option_no_beta_exp_replay.*Humanoid.*ensemble_off_policy.*', figure=0, average=True, max_timesteps=1e6, color=1)
    # plot(pattern='.*option_no_beta_exp_replay.*Humanoid.*on_policy.*', figure=0, average=True, max_timesteps=1e6, color=2)
    # plot(pattern='.*option_no_beta_exp_replay.*Humanoid.*half_off_policy.*', figure=0, average=True, max_timesteps=1e6, color=3)
    # plt.show()

    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1.*', figure=0)
    # plot(pattern='.*log/ensemble-RoboschoolHumanoid-v1.*', figure=1)
    # plot(pattern='.*log/ensemble-RoboschoolHumanoidFlagrun-v1.*', figure=2)
    # plot(pattern='.*log/ensemble-RoboschoolHumanoidFlagrunHarder-v1.*', figure=3)
    # plt.show()

    # plot(pattern='.*var_test_no_reward_scale.*', figure=0, average=True, max_timesteps=1e6, color=0)
    # plot(pattern='.*var_test_original.*', figure=0, average=True, max_timesteps=1e6, color=1)
    # plot(pattern='.*var_test_running_state.*', figure=0, average=True, max_timesteps=1e6, color=2)
    # plot(pattern='.*var_test_tanh-run.*', figure=0, average=True, max_timesteps=1e6, color=3)
    # plot(pattern='.*var_test_tanh_no_reward_scale-run.*', figure=0, average=True, max_timesteps=1e6, color=4)
    # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-RoboschoolHopper-v1/ddpg_continuous.*', figure=0, average=True, max_timesteps=1e6, color=5)
    # plt.show()

    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1/gamma_ddpg/mixed_target.*', figure=0)
    # plot(pattern='.*log/ensemble-RoboschoolAnt-v1/gamma_ddpg/vanilla_target.*', figure=1)
    # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-RoboschoolAnt-v1/ddpg_continuous.*', figure=2)
    # plt.show()

    # plot(pattern='.*ensemble_ddpg_exploration_0\.7.*', figure=0, average=True, max_timesteps=1e6, color=0)
    # plot(pattern='.*ensemble_ddpg_exploration_0\.3.*', figure=0, average=True, max_timesteps=1e6, color=5)
    # plot(pattern='.*ensemble_ddpg_constant_0\.3.*', figure=0, average=True, max_timesteps=1e6, color=6)
    # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-RoboschoolAnt-v1/ddpg_continuous.*', figure=0, average=True, max_timesteps=1e6, color=1)
    # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-RoboschoolAnt-v1/.*half_policy.*', figure=0, average=True, max_timesteps=1e6, color=2)
    # plot(pattern='.*log/option_no_beta_exp_replay/ensemble-RoboschoolAnt-v1/.*off_policy.*', figure=0, average=True, max_timesteps=1e6, color=3)
    # plt.show()

    # games = [
    #     'RoboschoolAnt-v1',
    #     'RoboschoolHopper-v1',
    #     'RoboschoolWalker2d-v1',
    #     'RoboschoolHalfCheetah-v1',
    #     'RoboschoolReacher-v1',
    #     'RoboschoolHumanoid-v1'
    # ]
    #
    # for i, game in enumerate(games):
    #     plot(pattern='.*log/option_no_beta_exp_replay/ensemble-%s.*original_ddpg.*' % (game), figure=i, average=True, max_timesteps=1e6, color=0)
    #     plot(pattern='.*log/option_no_beta_exp_replay/ensemble-%s.*half_policy.*' % (game), figure=i, average=True, max_timesteps=1e6, color=1)
    #     plot(pattern='.*log/option_no_beta_exp_replay/ensemble-%s.*on_policy.*' % (game), figure=i, average=True, max_timesteps=1e6, color=2)
    #     plot(pattern='.*log/option_no_beta_exp_replay/ensemble-%s.*off_policy.*' % (game), figure=i, average=True, max_timesteps=1e6, color=3)
    # plt.show()

    # patterns = ['.*var_test_constant_exploration_tanh.*',
    #             '.*var_test_no_reward_scale.*',
    #             '.*var_test_option_exploration_constant_tanh.*',
    #             '.*var_test_option_exploration_tanh.*',
    #             '.*var_test_original.*',
    #             '.*var_test_tanh.*']
    # for i, p in enumerate(patterns):
    #     # plot(pattern=p, figure=i, average=False, max_timesteps=1e6)
    #     plot(pattern=p, figure=0, color=i, average=True, max_timesteps=1e6)
    # plt.show()

    # ddpg_plot(pattern='.*log/ddpg_continuous.*', figure=0, x_interval=int(1e3), rep=20)
    # plt.show()

    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True
    }
    games = [
        'RoboschoolAnt-v1',
        'RoboschoolHopper-v1',
        'RoboschoolWalker2d-v1',
        'RoboschoolHalfCheetah-v1',
        'RoboschoolReacher-v1',
        'RoboschoolHumanoid-v1'
    ]
    # patterns = [
    #     'var_test_gaussian_no_reward_scale-run',
    #     'var_test_gaussian_tanh-run',
    #     'var_test_gaussian_tanh_no_reward_scale-run',
    #     'var_test_no_reward_scale-run',
    #     'var_test_original-run',
    #     'var_test_tanh-run',
    #     'var_test_tanh_no_reward_scale-run'
    # ]
    # patterns = [
    #     'original_ddpg',
    #     'off_policy',
    #     'on_policy',
    #     'half_policy'
    # ]
    # for i, game in enumerate(games):
    #     for j, p in enumerate(patterns):
    #         ddpg_plot(pattern='.*ddpg_ensemble_replay.*%s.*%s.*' % (game, p), figure=i, color=j, **kwargs)
    # plt.show()


    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True
    }
    # patterns = [
    #     'per_episode_decay',
    #     'per_episode_random',
    #     'per_step_decay',
    #     'per_step_random'
    # ]
    games = [
        'RoboschoolAnt-v1',
        'RoboschoolWalker2d-v1',
        'RoboschoolHopper-v1',
        'RoboschoolHalfCheetah-v1',
        'RoboschoolReacher-v1',
        'RoboschoolHumanoid-v1',
        'RoboschoolPong-v1',
        'RoboschoolHumanoidFlagrun-v1',
        'RoboschoolHumanoidFlagrunHarder-v1',
        'RoboschoolInvertedPendulum-v1',
        'RoboschoolInvertedPendulumSwingup-v1',
        'RoboschoolInvertedDoublePendulum-v1',
    ]
    patterns = [
        'd1m0',
        'd2m0',
        'shared'
    ]
    peaks = {}
    for j, game in enumerate(games):
        for i, p in enumerate(patterns):
            peak = ddpg_plot(pattern='.*DTreePG/plan-%s.*%s.*' % (game, p), figure=j, color=i, **kwargs)
        ddpg_plot(pattern='.*log/baseline-ddpg/baseline-%s.*baseline_ddpg.*' % (game), figure=j, color=i+1, **kwargs)
        ddpg_plot(pattern='.*log/baseline-ddpg/baseline-%s.*larger_ddpg.*' % (game), figure=j, color=i+2, **kwargs)

        plt.savefig('/home/shangtong/Documents/DTreePG/%s.png' % (game))
    # plt.show()
