import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_rl import *

FOLDER = '/Users/Shangtong/Dropbox/Paper/revisiting-residual/img'


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
            plt.plot(x, y, color=color, label=name if i == 0 else '')
    plt.legend()
    if 'y_lim' in kwargs.keys():
        plt.ylim(kwargs['y_lim'])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')


def ddpg_plot(**kwargs):
    kwargs.setdefault('average', True)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('max_x_len', None)
    kwargs.setdefault('type', 'mean')
    kwargs.setdefault('data', False)
    kwargs.setdefault('window', 0)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=kwargs['window'], max_timesteps=kwargs['max_timesteps'])
    if len(data) == 0:
        print('File not found')
        return
    data = [y[: len(y) // kwargs['rep'] * kwargs['rep']] for x, y in data]
    min_y = np.min([len(y) for y in data])
    data = [y[:min_y] for y in data]
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

    if kwargs['data']:
        return np.asarray([entry[1] for entry in data])

    print('')

    color = kwargs['color']
    if kwargs['average']:
        x = data[0][0]
        y = [entry[1] for entry in data]
        y = np.stack(y)
        if kwargs['type'] == 'mean':
            plotter.plot_standard_error(y, x, label=kwargs['label'], color=Plotter.COLORS[color])
        elif kwargs['type'] == 'median':
            plotter.plot_median_std(y, x, label=kwargs['label'], color=Plotter.COLORS[color])
        else:
            raise NotImplementedError
    else:
        for i, (x, y) in enumerate(data):
            plt.plot(x, y, color=Plotter.COLORS[color], label=names[i] if i == 0 else '')


def plot_ddpg_variants(type='mean'):
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True,
        'max_x_len': 101,
        'top_k': 0,
        'type': type,
        'window': 0,
    }

    game = 'dm-walker-stand'

    cfgs = ['bi', 'oo', 'tt', 'to', 'ot']
    titles = ['Bi-Res-DDPG', 'Res-DDPG', 'TT-Res-DDPG', 'TO-Res-DDPG', 'OT-Res-DDPG']

    patterns = [
        'net_cfg_%s-residual_0-run',
        'net_cfg_%s-residual_0\.05-run',
        'net_cfg_%s-residual_0\.1-run',
        'net_cfg_%s-residual_0\.2-run',
        'net_cfg_%s-residual_0\.4-run',
        'net_cfg_%s-residual_0\.8-run',
        'net_cfg_%s-residual_1\.0-run',
    ]

    labels = [
        r'$\eta=0$',
        r'$\eta=0.05$',
        r'$\eta=0.1$',
        r'$\eta=0.2$',
        r'$\eta=0.4$',
        r'$\eta=0.8$',
        r'$\eta=1$',
    ]

    l = len(cfgs)
    plt.figure(figsize=(l * 6, 5))
    plt.rc('text', usetex=True)
    plt.tight_layout()
    for i, cfg in enumerate(cfgs):
        plt.subplot(1, l, i + 1)
        for j, p in enumerate(patterns):
            ddpg_plot(pattern='.*residual-params/%s-%s.*' % (game, p % (cfg)), color=j, label=labels[j], **kwargs)
        plt.title(titles[i], fontsize=30, fontweight="bold")
        plt.ylim([0, 1000])
        plt.xticks([0, int(1e6)], ['0', r'$10^6$'])
        plt.tick_params(axis='x', labelsize=30)
        plt.tick_params(axis='y', labelsize=30)
        plt.xlabel('Steps', fontsize=30)
        if i == 2:
            plt.legend(fontsize=17, frameon=False)
        if not i:
            plt.ylabel('Episode Return', fontsize=30)
        else:
            plt.tick_params(labelleft=False)
    plt.savefig('%s/ddpg-variants-%s.pdf' % (FOLDER, type), bbox_inches='tight')
    plt.show()


def extract_auc_data():
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True,
        'max_x_len': 101,
        'top_k': 0,
    }

    games = [
        'dm-acrobot-swingup',
        'dm-acrobot-swingup_sparse',
        'dm-ball_in_cup-catch',
        'dm-cartpole-swingup',
        'dm-cartpole-swingup_sparse',
        'dm-cartpole-balance',
        'dm-cartpole-balance_sparse',
        'dm-cheetah-run',
        'dm-finger-turn_hard',
        'dm-finger-spin',
        'dm-finger-turn_easy',
        'dm-fish-upright',
        'dm-fish-swim',
        'dm-hopper-stand',
        'dm-hopper-hop',
        'dm-humanoid-stand',
        'dm-humanoid-walk',
        'dm-humanoid-run',
        'dm-manipulator-bring_ball',
        'dm-pendulum-swingup',
        'dm-point_mass-easy',
        'dm-reacher-easy',
        'dm-reacher-hard',
        'dm-swimmer-swimmer15',
        'dm-swimmer-swimmer6',
        'dm-walker-stand',
        'dm-walker-walk',
        'dm-walker-run',
    ]

    patterns = [
        'remark_residual-residual_0\.05-target_net_residual_True-run',
        'remark_residual-residual_0-target_net_residual_True-run',
    ]

    names = []
    improvements = []
    for game in games:
        AUC = []
        for p in patterns:
            data = ddpg_plot(pattern='.*residual-ddpg/%s-%s.*' % (game, p), data=True, **kwargs)
            AUC.append(data.mean(0).sum())
        improvements.append((AUC[0] - AUC[1]) / AUC[1])
        names.append(game[3:])
        print(names[-1], improvements[-1])

    with open('./data/residual/auc.bin', 'wb') as f:
        pickle.dump([names, improvements], f)


def plot_auc_improvements():
    with open('./data/residual/auc.bin', 'rb') as f:
        games, improvements = pickle.load(f)

    indices = list(reversed(np.argsort(improvements)))
    games = [games[i] for i in indices]
    improvements = [improvements[i] for i in indices]

    print(np.median(improvements), np.mean(improvements))

    for g, i in zip(games, improvements):
        print(g, i)

    x = np.arange(len(improvements))

    plt.tight_layout()
    plt.bar(x, improvements)
    plt.xticks(x, games, rotation=-90)
    plt.gca().invert_yaxis()
    yticks = np.arange(-1, 4, 1)
    plt.yticks(yticks, ['-100%', '0', '100%', '200%', '300%'], rotation=-90, verticalalignment='center')
    # plt.ylabel('AUC Improvement', rotation=-90)
    plt.savefig('%s/ddpg-auc.png' % (FOLDER), bbox_inches='tight')
    plt.show()


def plot_oracle(type='mean'):
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True,
        'max_x_len': 101,
        'top_k': 0,
        'type': type,
    }
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Humanoid-v2',
    ]

    patterns = [
        'action_noise_0\.1-live_action_False-plan_steps_1-residual_0-target_net_residual_True-run',
        'action_noise_0\.1-live_action_False-plan_steps_1-residual_0\.2-target_net_residual_False-run',
        'MVE_3-plan_False-run',
    ]

    labels = [
        'Dyna-DDPG',
        'Res-Dyna-DDPG',
        'MVE-DDPG',
    ]

    l = len(games)
    plt.figure(figsize=(l * 6, 5))
    plt.rc('text', usetex=True)
    plt.tight_layout()
    for j, game in enumerate(games):
        plt.subplot(1, l, j + 1)
        ddpg_plot(pattern='.*mujoco-baseline/%s-%s.*' % (game, 'remark_ddpg-run'), color=0, label='DDPG', **kwargs)
        for i, p in enumerate(patterns):
            ddpg_plot(pattern='.*oracle-ddpg/%s-%s.*' % (game, p), color=i + 1, label=labels[i], **kwargs)
        plt.title(game, fontsize=30, fontweight="bold")
        plt.xticks([0, int(1e6)], ['0', r'$10^6$'])
        plt.tick_params(axis='x', labelsize=30)
        plt.tick_params(axis='y', labelsize=25)
        plt.xlabel('Steps', fontsize=30)
        if not j:
            plt.ylabel('Episode Return', fontsize=30)
            plt.legend(fontsize=17, frameon=False)
    plt.savefig('%s/ddpg-oracle-%s.png' % (FOLDER, type), bbox_inches='tight')
    plt.show()


def plot_dyna(type='mean'):
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True,
        'max_x_len': 101,
        'top_k': 0,
        'type': type,
    }
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Humanoid-v2',
    ]

    patterns = [
        'action_noise_0.1-plan_steps_1-residual_0-skip_False-target_net_residual_True-run',
        'action_noise_0.1-plan_steps_1-residual_0\.2-skip_False-target_net_residual_False-run',
        'MVE_3-plan_False-skip_False-run',
    ]

    labels = [
        'Dyna-DDPG',
        'Res-Dyna-DDPG',
        'MVE-DDPG',
    ]

    l = len(games)
    plt.figure(figsize=(l * 6, 5))
    plt.rc('text', usetex=True)
    plt.tight_layout()
    for j, game in enumerate(games):
        plt.subplot(1, l, j + 1)
        if j == 0:
            label = None
        else:
            label = 'DDPG'
        ddpg_plot(pattern='.*mujoco-baseline/%s-%s.*' % (game, 'remark_ddpg-run'), color=0, label=label, **kwargs)
        for i, p in enumerate(patterns):
            if j == 0:
                label = None
            else:
                label = labels[i]
            ddpg_plot(pattern='.*mve-ddpg/%s-%s.*' % (game, p), color=i + 1, label=label, **kwargs)
            ddpg_plot(pattern='.*dyna-ddpg-main/%s-%s.*' % (game, p), color=i + 1, label=label, **kwargs)
        if j == 0:
            p = 'action_noise_0\.1-plan_steps_1-residual_0\.05-skip_False-target_net_residual_False-run'
            label = r'Res-Dyna-DDPG($\eta=0.05$)'
            ddpg_plot(pattern='.*dyna-ddpg-extra/%s-%s.*' % (game, p), color=i + 2, label=label, **kwargs)
        plt.title(game, fontsize=30, fontweight="bold")
        plt.xticks([0, int(1e6)], ['0', r'$10^6$'])
        plt.tick_params(axis='x', labelsize=30)
        plt.tick_params(axis='y', labelsize=25)
        plt.xlabel('Steps', fontsize=30)
        if not j:
            plt.ylabel('Episode Return', fontsize=30)
        if j == 1 or j == 0:
            plt.legend(fontsize=17, frameon=False)
    plt.savefig('%s/ddpg-dyna-%s.png' % (FOLDER, type), bbox_inches='tight')
    plt.show()


def plot_mf_ddpg(type='mean'):
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True,
        'max_x_len': 101,
        'top_k': 0,
        'type': type,
    }

    games = [
        'dm-acrobot-swingup',
        'dm-acrobot-swingup_sparse',
        'dm-ball_in_cup-catch',
        'dm-cartpole-swingup',
        'dm-cartpole-swingup_sparse',
        'dm-cartpole-balance',
        'dm-cartpole-balance_sparse',
        'dm-cheetah-run',
        'dm-finger-turn_hard',
        'dm-finger-spin',
        'dm-finger-turn_easy',
        'dm-fish-upright',
        'dm-fish-swim',
        'dm-hopper-stand',
        'dm-hopper-hop',
        'dm-humanoid-stand',
        'dm-humanoid-walk',
        'dm-humanoid-run',
        'dm-manipulator-bring_ball',
        'dm-pendulum-swingup',
        'dm-point_mass-easy',
        'dm-reacher-easy',
        'dm-reacher-hard',
        'dm-swimmer-swimmer15',
        'dm-swimmer-swimmer6',
        'dm-walker-stand',
        'dm-walker-walk',
        'dm-walker-run',
    ]

    patterns = [
        'remark_residual-residual_0-target_net_residual_True-run',
        'remark_residual-residual_0\.05-target_net_residual_True-run',
    ]

    labels = [
        'DDPG',
        r'Bi-Res-DDPG($\eta=0.05$)',
    ]

    l = len(games)
    n_col = 4
    n_row = l // n_col
    width = 5

    plt.figure(figsize=(n_col * width, n_row * width))
    plt.rc('text', usetex=True)
    plt.tight_layout()
    for j, game in enumerate(games):
        plt.subplot(n_row, n_col, j + 1)
        for i, p in enumerate(patterns):
            ddpg_plot(pattern='.*dm-residual-ddpg/%s-%s.*' % (game, p), color=i, label=labels[i], **kwargs)
        title = game[3:].replace('_', '\\_')
        plt.title(title, fontsize=30, fontweight="bold")
        plt.tick_params(axis='x', labelsize=30)
        plt.tick_params(axis='y', labelsize=20)
        if not j:
            plt.legend(fontsize=17, frameon=False)
        if j % n_col == 0:
            plt.ylabel('Episode Return', fontsize=30)

        if j >= l - n_col:
            plt.xlabel('Steps', fontsize=30)
            plt.xticks([0, int(1e6)], ['0', r'$10^6$'])
        else:
            plt.tick_params(labelbottom=False)

    plt.savefig('%s/ddpg-mf-%s.png' % (FOLDER, type), bbox_inches='tight')
    plt.show()


def plot_rebuttal():
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
        'Swimmer-v2',
        'Humanoid-v2',
    ]

    patterns = [
        'remark_ddpg-run',
        'remark_residual-residual_0\.05-target_net_residual_True-run',
    ]

    labels = [
        'DDPG',
        r'Bi-Res-DDPG($\eta=0.05$)',
    ]

    l = len(games)
    plt.figure(figsize=(l * 6, 5))
    plt.rc('text', usetex=True)
    plt.tight_layout()
    for j, game in enumerate(games):
        plt.subplot(1, l, j + 1)
        for i, p in enumerate(patterns):
            ddpg_plot(pattern='.*residual-ddpg/%s-%s.*' % (game, p), color=i, name=game, label=labels[i], **kwargs)
            ddpg_plot(pattern='.*mujoco-baseline/%s-%s.*' % (game, p), color=i, name=game, label=labels[i], **kwargs)
        if j == 0:
            plt.legend()
        plt.title(game, fontsize=30, fontweight="bold")
        plt.xticks([0, int(1e6)], ['0', r'$10^6$'])
        plt.tick_params(axis='x', labelsize=30)
        plt.tick_params(axis='y', labelsize=25)
        plt.xlabel('Steps', fontsize=30)
        if not j:
            plt.ylabel('Episode Return', fontsize=30)
    plt.savefig('%s/rebuttal.pdf' % (FOLDER), bbox_inches='tight')
    plt.show()


def extract_auc_data_mujoco():
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
        'Swimmer-v2',
        'Humanoid-v2',
    ]

    patterns = [
        'remark_residual-residual_0\.05-target_net_residual_True-run',
        'remark_ddpg-run',
    ]

    names = []
    improvements = []
    for game in games:
        AUC = []
        for i, p in enumerate(patterns):
            if i == 0:
                data = ddpg_plot(pattern='.*residual-ddpg/%s-%s.*' % (game, p), data=True, **kwargs)
            else:
                data = ddpg_plot(pattern='.*mujoco-baseline/%s-%s.*' % (game, p), data=True, **kwargs)
            AUC.append(data.mean(0).sum())
        improvements.append((AUC[0] - AUC[1]) / AUC[1])
        names.append(game)
        print(names[-1], improvements[-1])

    with open('./data/residual/auc_mujoco.bin', 'wb') as f:
        pickle.dump([names, improvements], f)


def plot_auc_improvements_mujoco():
    with open('./data/residual/auc_mujoco.bin', 'rb') as f:
        games, improvements = pickle.load(f)

    indices = list(reversed(np.argsort(improvements)))
    games = [games[i] for i in indices]
    improvements = [improvements[i] for i in indices]

    print(np.median(improvements), np.mean(improvements))

    for g, i in zip(games, improvements):
        print(g, i)

    x = np.arange(len(improvements))

    plt.tight_layout()
    plt.bar(x, improvements)
    plt.xticks(x, games, rotation=-90)
    plt.gca().invert_yaxis()
    yticks = np.arange(-1, 4, 1)
    plt.yticks(yticks, ['-100%', '0', '100%', '200%', '300%'], rotation=-90, verticalalignment='center')
    # plt.ylabel('AUC Improvement', rotation=-90)
    plt.savefig('%s/ddpg-auc_mujoco.pdf' % (FOLDER), bbox_inches='tight')
    plt.show()


# def plot_mf_dqn():
#     train_kwargs = {
#         'episode_window': 100,
#         'top_k': 0,
#         'max_timesteps': int(2e7),
#         'average': False,
#         'x_interval': 100
#     }
#
#     games = [
#         'BeamRiderNoFrameskip-v4',
#         'SeaquestNoFrameskip-v4',
#         'BreakoutNoFrameskip-v4',
#         # 'PongNoFrameskip-v4',
#         # 'QbertNoFrameskip-v4',
#         # 'SpaceInvadersFrameskip-v4',
#     ]
#
#     patterns = [
#         'residual_0-target_net_residual_True-run',
#         'r_aware_False-residual_0\.05-target_net_residual_True-run',
#         'r_aware_False-residual_1-target_net_residual_True-run',
#     ]
#
#     labels = [
#         'DQN',
#         r'Bi-Res-DQN($\eta=0.05$)',
#         r'Bi-Res-DQN($\eta=1$)',
#     ]
#
#     l = len(games)
#     plt.figure(figsize=(l * 5, 5))
#     plt.rc('text', usetex=True)
#     plt.tight_layout()
#     for j, game in enumerate(games):
#         plt.subplot(1, l, j + 1)
#         for i, p in enumerate(patterns):
#             plot(pattern='.*residual-dqn/.*%s.*%s.*' % (game, p), **train_kwargs, figure=j, color=i)
#
#         title = game[3:].replace('_', '\\_')
#         plt.title(title, fontsize=30, fontweight="bold")
#         plt.tick_params(axis='x', labelsize=30)
#         plt.tick_params(axis='y', labelsize=20)
#         if not j:
#             plt.legend(fontsize=17, frameon=False)
#         if j % n_col == 0:
#             plt.ylabel('Episode Return', fontsize=30)
#
#         if j >= l - n_col:
#             plt.xlabel('Steps', fontsize=30)
#             plt.xticks([0, int(1e6)], ['0', r'$10^6$'])
#         else:
#             plt.tick_params(labelbottom=False)
#
#     plt.show()


if __name__ == '__main__':
    # extract_auc_data()
    # plot_ddpg_variants(type='mean')
    # plot_ddpg_variants(type='median')
    # plot_auc_improvements()
    # plot_oracle(type='mean')
    # plot_oracle(type='median')
    # plot_dyna(type='mean')
    # plot_dyna(type='median')
    # plot_mf_ddpg(type='mean')
    # plot_mf_ddpg(type='median')
    # plot_rebuttal()
    # extract_auc_data_mujoco()
    plot_auc_improvements_mujoco()
