import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_rl import *

def plot(**kwargs):
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    # kwargs.setdefault('top_k_perf', lambda x: np.mean(x[-20:]))
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('episode_window', 100)
    kwargs.setdefault('x_interval', 1000)
    kwargs.setdefault('down_sample', False)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=kwargs['episode_window'], max_timesteps=kwargs['max_timesteps'])
    print('')

    if kwargs['average']:
        color = kwargs['color']
        x, y = plotter.average(data, kwargs['x_interval'], kwargs['max_timesteps'], top_k=kwargs['top_k'])
        print(y.shape)
        if kwargs['down_sample']:
            indices = np.linspace(0, len(x) - 1, 500).astype(np.int)
            x = x[indices]
            y = y[:, indices]
        name = names[0].split('/')[-1]
        plotter.plot_standard_error(y, x, label=name, color=Plotter.COLORS[color])
        # sns.tsplot(y, x, condition=name, , ci='sd')
        plt.title(names[0])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            if 'color' not in kwargs.keys():
                color = Plotter.COLORS[i]
            else:
                color = Plotter.COLORS[kwargs['color']]
            plt.plot(x, y, color=color, label=name if i==0 else '')
    plt.legend()
    if 'y_lim' in kwargs.keys():
        plt.ylim(kwargs['y_lim'])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')

def plot_atari():
    train_kwargs = {
        'episode_window': 100,
        'top_k': 0,
        'max_timesteps': int(2e7),
        # 'max_timesteps': int(3e7),
        'average': False,
        'x_interval': 100
    }

    games = ['Breakout', 'Alien']

    patterns = [
        # 'mix_nq_aux_r5',
        'mix_nq_rmix_r5',
        'mix_nq_study_r5'
    ]

    l = len(games)
    plt.figure(figsize=(l * 10, 10))
    for j, game in enumerate(games):
        plt.subplot(1, l, j + 1)
        for i, p in enumerate(patterns):
            plot(pattern='.*rmix/.*%s.*%s.*' % (game, p), **train_kwargs, figure=j, color=i)
    plt.show()

def ddpg_plot(**kwargs):
    kwargs.setdefault('average', True)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('max_x_len', None)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=kwargs['max_timesteps'])
    if len(data) == 0:
        print('File not found')
        return
    data = [y[: len(y) // kwargs['rep'] * kwargs['rep']] for x, y in data]
    min_y = np.min([len(y) for y in data])
    data = [y[ :min_y] for y in data]
    new_data = []
    for y in data:
        y = np.reshape(np.asarray(y), (-1, kwargs['rep'])).mean(-1)
        x = np.arange(y.shape[0]) * kwargs['x_interval']
        max_x_len = kwargs['max_x_len']
        if max_x_len is not None:
            x = x[:max_x_len]
            y = y[:max_x_len]
        new_data.append([x, y])
    data = new_data

    if kwargs['top_k']:
        scores = []
        for x, y in data:
            scores.append(np.sum(y))
        best = list(reversed(np.argsort(scores)))
        best = best[:kwargs['top_k']]
        data = [data[i] for i in best]


    print('')

    game = kwargs['name']
    color = kwargs['color']
    if kwargs['average']:
        x = data[0][0]
        y = [entry[1] for entry in data]
        y = np.stack(y)
        name = names[0].split('/')[-1]
        plotter.plot_standard_error(y, x, label=name, color=Plotter.COLORS[color])
        plt.title(game)
    else:
        for i, (x, y) in enumerate(data):
            plt.plot(x, y, color=Plotter.COLORS[color], label=names[i] if i==0 else '')
    plt.legend()
    # plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')

def plot_mujoco():
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True,
        'max_x_len': 101,
        'top_k': 0,
    }
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        # 'Reacher-v2',
        'Swimmer-v2',
        'Humanoid-v2',
    ]
    # games = ['RoboschoolHumanoid-v1', 'RoboschoolAnt-v1', 'RoboschoolHumanoidFlagrun-v1', 'RoboschoolHumanoidFlagrunHarder-v1']
    # games = ['RoboschoolHumanoid-v1']
    # games = ['RoboschoolAnt-v1']
    # games = ['Ant-v2', 'HumanoidStandup-v2']
    games = ['Swimmer-v2']

    patterns = [
        # 'remark_ddpg-run',
        # 'action_noise_0\.1-max_uncertainty_1-random_t_mask_False-run',
        # 'action_noise_0\.1-max_uncertainty_1-random_t_mask_True-run',
        # 'action_noise_0-max_uncertainty_1-random_t_mask_False-run',
        # 'action_noise_0-max_uncertainty_inf-run',
        # 'action_noise_0\.1-max_uncertainty_inf-run',

        # 'action_noise_0\.2-max_uncertainty_1-random_t_mask_False-run',
        # 'action_noise_0\.05-max_uncertainty_1-random_t_mask_False-run',

        # 'action_noise_0\.1-live_action_False-max_uncertainty_1-plan_steps_1-run',
        # 'action_noise_0\.1-live_action_False-max_uncertainty_1-plan_actor_True-plan_steps_1-run',
        # 'action_noise_0\.1-live_action_False-max_uncertainty_1-plan_steps_2-run',
        # 'action_noise_0\.1-live_action_False-max_uncertainty_1-plan_steps_4-run',

        # 'action_noise_0\.1-live_action_False-max_uncertainty_2-plan_steps_1-run',
        # 'action_noise_0\.1-live_action_False-max_uncertainty_4-plan_steps_1-run',

        # 'action_noise_0.1-live_action_True-max_uncertainty_1-plan_steps_1-run',

        # 'action_noise_0\.1-live_action_False-max_uncertainty_2-model_agg_max-plan_steps_2-run',
        # 'action_noise_0\.1-live_action_False-max_uncertainty_2-model_agg_min-plan_steps_2-run',
        # 'action_noise_0\.1-live_action_False-max_uncertainty_2-model_agg_mean-plan_steps_2-run',

        # 'action_noise_0\.1-live_action_False-max_uncertainty_2-model_agg_mean-plan_steps_1-run',
        # 'action_noise_0\.05-live_action_False-max_uncertainty_2-model_agg_mean-plan_steps_1-run',
        # 'action_noise_0\.2-live_action_False-max_uncertainty_2-model_agg_mean-plan_steps_1-run',
        # 'action_noise_0\.1-live_action_False-max_uncertainty_1-model_agg_mean-plan_steps_2-run',

        # 'action_noise_0-live_action_False-max_uncertainty_2-plan_actor_True-plan_steps_1-state_noise_0\.2-run',
        # 'action_noise_0-live_action_False-max_uncertainty_2-plan_actor_True-plan_steps_1-state_noise_0\.05-run',
        # 'action_noise_0-live_action_False-max_uncertainty_2-plan_actor_True-plan_steps_1-state_noise_0\.1-run',

        # 'action_noise_0-live_action_False-max_uncertainty_2-plan_actor_False-plan_steps_1-state_noise_0\.2-run',
        # 'action_noise_0-live_action_False-max_uncertainty_2-plan_actor_False-plan_steps_1-state_noise_0\.05-run',
        # 'action_noise_0-live_action_False-max_uncertainty_2-plan_actor_False-plan_steps_1-state_noise_0\.1-run',

        # 'action_noise_0-live_action_True-max_uncertainty_2-plan_actor_True-plan_steps_1-state_noise_0\.2-run',
        # 'action_noise_0-live_action_True-max_uncertainty_2-plan_actor_True-plan_steps_1-state_noise_0\.05-run',
        # 'action_noise_0-live_action_True-max_uncertainty_2-plan_actor_True-plan_steps_1-state_noise_0\.1-run',

        # 'action_noise_0\.1-live_action_False-max_uncertainty_1-model_agg_mean-plan_actor_True-plan_steps_2-run',
        # 'action_noise_0\.2-live_action_False-max_uncertainty_1-model_agg_mean-plan_actor_True-plan_steps_2-run',
        # 'action_noise_0\.1-live_action_False-max_uncertainty_2-model_agg_mean-plan_actor_True-plan_steps_1-run',
        # 'action_noise_0\.2-live_action_False-max_uncertainty_2-model_agg_mean-plan_actor_True-plan_steps_1-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-run',
        # 'action_noise_0\.1-live_action_True-plan_steps_1-run',
        # 'action_noise_0\.2-live_action_False-plan_steps_1-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_2-run',
        # 'plan_False-run',
        # 'plan_False-real_updates_2-run',
        # 'action_noise_0\.1-live_action_False-plan_actor_True-plan_steps_1-residual_False-run',

        # 'action_noise_0\.1-live_action_False-plan_actor_False-plan_steps_1-residual_True-run',
        # 'action_noise_0\.1-live_action_False-plan_actor_True-plan_steps_1-residual_True-run',


        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.1-target_net_residual_False-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.5-target_net_residual_False-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_1\.0-target_net_residual_False-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.1-target_net_residual_True-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.5-target_net_residual_True-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_1\.0-target_net_residual_True-run',

        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0-target_net_residual_False-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0-target_net_residual_True-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.05-target_net_residual_False-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.05-target_net_residual_True-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.2-target_net_residual_False-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.2-target_net_residual_True-run',

        # 'action_noise_0\.1-live_action_False-plan_steps_1-prediction_noise_0\.01-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-prediction_noise_0\.05-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-prediction_noise_0\.1-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-prediction_noise_0\.2-run',

        # 'residual_0\.01-run',
        # 'residual_0\.1-run',
        # 'residual_0\.5-run',
        # 'residual_1-run',

        # learned model
        # 'action_noise_0.1-plan_steps_1-residual_0-skip_False-target_net_residual_True-run',
        # 'action_noise_0.1-plan_steps_1-residual_0\.2-skip_False-target_net_residual_False-run',

        # oralce model
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0-target_net_residual_True-run',
        # 'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.2-target_net_residual_False-run',

        # 'action_noise_0.1-plan_steps_1-residual_0\.05-skip_False-target_net_residual_True-run',
        # 'action_noise_0.1-plan_steps_1-residual_0\.1-skip_False-target_net_residual_True-run',
        # 'action_noise_0.1-plan_steps_1-residual_0\.05-skip_False-target_net_residual_False-run',
        # 'action_noise_0.1-plan_steps_1-residual_0\.1-skip_False-target_net_residual_False-run',

        # 'remark_residual-residual_0\.05-target_net_residual_False-run',
        # 'remark_residual-residual_0\.1-target_net_residual_False-run',
        # 'remark_residual-residual_0\.2-target_net_residual_False-run',
        # 'remark_residual-residual_0\.4-target_net_residual_False-run',
        # 'remark_residual-residual_0\.8-target_net_residual_False-run',
        # 'remark_residual-residual_1-target_net_residual_False-run',

        'remark_residual-residual_0\.05-target_net_residual_True-run',
        'remark_residual-residual_0\.1-target_net_residual_True-run',
        'remark_residual-residual_0\.2-target_net_residual_True-run',
        'remark_residual-residual_0\.4-target_net_residual_True-run',
        'remark_residual-residual_0\.8-target_net_residual_True-run',
        'remark_residual-residual_1-target_net_residual_True-run',
    ]

    l = len(games)
    plt.figure(figsize=(l * 15, 15))
    for j, game in enumerate(games):
        plt.subplot(1, l, j+1)
        ddpg_plot(pattern='.*mujoco-baseline/%s-%s.*' % (game, 'remark_ddpg-run'), color=0, name=game, **kwargs)
        for i, p in enumerate(patterns):
            # ddpg_plot(pattern='.*oracle-ddpg/%s-%s.*' % (game, p), color=i, name=game, **kwargs)
            # ddpg_plot(pattern='.*dyna-ddpg/%s-%s.*' % (game, p), color=i, name=game, **kwargs)
            ddpg_plot(pattern='.*residual-ddpg/%s-%s.*' % (game, p), color=i+1, name=game, **kwargs)
    plt.show()

if __name__ == '__main__':
    plot_mujoco()