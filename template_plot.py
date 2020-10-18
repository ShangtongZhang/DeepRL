import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
from deep_rl import *


def plot_ppo():
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Reacher-v2',
    ]

    patterns = [
        'remark_ppo',
    ]

    labels = [
        'PPO'
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/benchmark/ppo',
                       interpolation=100,
                       window=0,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/PPO.png', bbox_inches='tight')


def plot_ddpg_td3():
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Reacher-v2',
    ]

    patterns = [
        'remark_ddpg',
        'remark_td3',
    ]

    labels = [
        'DDPG',
        'TD3',
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TEST,
                       root='./data/benchmark',
                       interpolation=0,
                       window=0,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/mujoco_eval.png', bbox_inches='tight')


def plot_atari():
    plotter = Plotter()
    games = [
        'BreakoutNoFrameskip-v4',
    ]

    patterns = [
        'remark_a2c',
        'remark_categorical',
        'remark_dqn',
        'remark_n_step_dqn',
        'remark_option_critic',
        'remark_ppo',
        'remark_quantile',
    ]

    labels = [
        'A2C',
        'C51',
        'DQN',
        'N-Step DQN',
        'OC',
        'PPO',
        'QR-DQN',
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=100,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/benchmark/atari',
                       interpolation=0,
                       window=100,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/Breakout.png', bbox_inches='tight')


def plot_reverse_td_lambda(metric):
    log_root = './log/reverse-rl/reverse-td-lambda'
    # log_root = './tf_log/reverse-td-lambda'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/reverse-rl/img'
    plotter = Plotter()
    game = 'RobotTabular-v0'

    def score(y):
        print(y.shape)
        assert y.shape[0] == 30
        if metric == 'auc':
            return -np.mean(y)
        elif metric == 'asym':
            return -np.mean(y[:, -1:])
        else:
            raise NotImplementedError

    # lams = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    lams = [0, 0.3, 0.7, 0.9, 1.0]
    lrs = [1e-3, 5e-3, 1e-2, 5e-2]
    all_labels = []
    all_patterns = []

    for lam in lams:
        best = []
        algos = ['ReverseTD']
        for i, algo in enumerate(algos):
            ids = []
            labels = []
            patterns = []

            for lr in lrs:
                template = '.*%s-lam_%s-loss_type_td-lr_%s-mu_0\.5-pi_0\.5-run.*' % (game, lam, lr)
                ids.append(lr)
                # label = r'%s($\alpha=3^{%s}$)' % (algo, round(math.log(lr, 3)))
                label = r'$\lambda = %s (\alpha = %s)$' % (lam, lr)
                labels.append(label)
                patterns.append(template)

            indices = plotter.select_best_parameters(
                patterns, score=score, root=log_root, tag='v_bar_loss', right_align=True, right_most=0)
            best.append([ids[indices[0]], labels[indices[0]], patterns[indices[0]]])

        for _, label, pattern in best:
            all_patterns.append(pattern)
            all_labels.append(label)

    titles = ['ReverseTD']
    all_patterns = [all_patterns]
    all_labels = [all_labels]

    fontsize = 22

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        plt.figure(figsize=(l * 5, 5))
        for i, title in enumerate(titles):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(all_patterns[i]):
                label = all_labels[i][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern=p, **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se')
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.xlabel('Steps', fontsize=fontsize)
            plt.ylim([0, 1])
            plt.yticks([0, 1], [0, 1], fontsize=fontsize)
            plt.xticks([0, 50], ['0', r'$5 \times 10^4$'], fontsize=fontsize)
            # plt.tick_params(axis='y', labelsize=fontsize)
            plt.ylabel('MVE', rotation='horizontal', fontsize=fontsize)
            plt.title(title, fontsize=fontsize)
            plt.legend(fontsize=12)

    plot_games(plotter,
               titles=titles,
               agg='mean',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='v_bar_loss',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/reverse-td-lambda-%s.pdf' % (img_root, metric), bbox_inches='tight')


def plot_reverse_td_mutation_robot():
    log_root = './log/reverse-rl/mutation'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/reverse-rl/img'
    plotter = Plotter()
    game_index = 0
    games = ['RobotTabular-v0']
    game = games[game_index]

    titles = ['Reward Anomaly', 'Policy Anomaly']
    all_labels = [[
        '2'
    ], [
        '0.9'
    ]]
    if game_index == 0:
        all_patterns = [[
            '.*logger-%s-frozen_True-loss_type_qr-lr_0\.01-max_steps_20000-mu_0\.1-mutation_1-num_quantiles_20-pi_0\.1-run' % (
                game),
        ], [
            '.*logger-%s-frozen_True-loss_type_qr-lr_0\.01-max_steps_20000-mu_0\.1-mutation_2-num_quantiles_20-pi_0\.1-run' % (
                game),
        ]]
        y_ticks = [[0, 1], [0, 1]]
    elif game_index == 1:
        all_patterns = [[
            '.*logger-Reacher-v3-frozen_True-loss_type_qr-max_steps_20000-mu_0\.1-mutation_1-num_quantiles_10-pi_0\.1-run-.*'
        ], [
            '.*logger-Reacher-v3-frozen_True-loss_type_qr-max_steps_20000-mu_0\.1-mutation_2-num_quantiles_10-pi_0\.1-run-.*'
        ]]
        y_ticks = [[0.5, 1], [0.5, 1]]
    else:
        raise NotImplementedError

    fontsize = 22

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        plt.figure(figsize=(l * 5, 5))
        for i, title in enumerate(titles):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(all_patterns[i]):
                label = all_labels[i][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern=p, **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                print(x.shape, y.shape)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se')
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.xlabel('Steps', fontsize=fontsize)
            plt.xticks([0, 50, int(1e2)], ['0', r'$10^4$', r'$2 \times 10^4$'], fontsize=fontsize)
            plt.yticks(y_ticks[0], y_ticks[1], fontsize=fontsize)
            # plt.tick_params(axis='y', labelsize=fontsize)
            if not i:
                # plt.ylabel('Prob(Anomaly)', horizontalalignment='right', rotation='horizontal', fontsize=fontsize)
                plt.ylabel('Prob(Anomaly)', fontsize=fontsize)
                # plt.ylabel(r'MSE$(\tau)$', fontsize=fontsize)
            plt.title(title, fontsize=fontsize)
            plt.legend(fontsize=fontsize)

    plot_games(plotter,
               titles=titles,
               agg='mean',
               downsample=100,
               right_align=True,
               right_most=0,
               tag='prob_outlier',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/%s-anomaly.pdf' % (img_root, game), bbox_inches='tight')


def plot_reverse_td_mutation_mujoco(game_index):
    log_root = './log/reverse-rl/mutation'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/reverse-rl/img'
    plotter = Plotter()
    games = ['Reacher-v3']
    game = games[game_index]

    mutation_metas = {1: [-1, -5, -10], 2: [0.9, 1.8, 2.7]}

    titles = ['Reward Anomaly', 'Policy Anomaly']
    all_labels = []
    all_patterns = []
    for i in range(1, 3):
        all_patterns.append([])
        all_labels.append([])
        for meta in mutation_metas[i]:
            all_patterns[-1].append('.*Reacher-v3-frozen_True-loss_type_qr-max_steps_20000-mu_0.1-mutation_%s-mutation_meta_%s-pi_0.1-run-.*' % (i, meta))
            all_labels[-1].append('%s' % (meta))

    fontsize = 22

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        plt.figure(figsize=(l * 5, 5))
        for i, title in enumerate(titles):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(all_patterns[i]):
                label = all_labels[i][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern=p, **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                print(x.shape, y.shape)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se')
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.xlabel('Steps', fontsize=fontsize)
            plt.xticks([0, 50, 100], ['0', r'$10^4$', r'$2 \times 10^4$'], fontsize=fontsize)
            plt.yticks([0.5, 1], [0.5, 1], fontsize=fontsize)
            plt.ylim([0.5, 1])
            # plt.tick_params(axis='y', labelsize=fontsize)
            if not i:
                # plt.ylabel('Prob(Anomaly)', horizontalalignment='right', rotation='horizontal', fontsize=fontsize)
                plt.ylabel('Prob(Anomaly)', fontsize=fontsize)
                # plt.ylabel(r'MSE$(\tau)$', fontsize=fontsize)
            plt.title(title, fontsize=fontsize)
            plt.legend(fontsize=fontsize)

    plot_games(plotter,
               titles=titles,
               agg='mean',
               downsample=100,
               right_align=True,
               right_most=0,
               tag='prob_outlier',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/%s-anomaly.pdf' % (img_root, game), bbox_inches='tight')


def plot_reverse_td_robot_tasks_train():
    log_root = './log/reverse-rl/mutation'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/reverse-rl/img'
    plotter = Plotter()
    game = 'RobotTabular-v0'

    titles = ['Off-policy QR-Reverse-TD']
    all_labels = [[
        'Off-policy QR-Reverse-TD',
    ]]
    all_patterns = [[
        '.*logger-RobotTabular-v0-loss_type_qr-lr_0\.01-max_steps_300000-mu_0\.5-mutation_2-num_quantiles_20-pi_0\.1-run-.*'
    ]]

    fontsize = 22

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        plt.figure(figsize=(l * 5, 5))
        for i, title in enumerate(titles):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(all_patterns[i]):
                label = all_labels[i][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern=p, **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                print(x.shape, y.shape)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se')
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.xlabel('Steps', fontsize=fontsize)
            plt.xticks([0, 300], ['0', r'$3 \times 10^5$'], fontsize=fontsize)
            plt.yticks([0, 7], [0, 7], fontsize=fontsize)
            # plt.tick_params(axis='y', labelsize=fontsize)
            if not i:
                # plt.ylabel('MVE', rotation='horizontal', fontsize=fontsize)
                plt.ylabel('MVE', fontsize=fontsize)
            plt.title(title, fontsize=fontsize)
            # plt.legend(fontsize=fontsize)

    plot_games(plotter,
               titles=titles,
               agg='mean_std',
               downsample=100,
               right_align=True,
               right_most=0,
               tag='v_bar_loss',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/%s-qr-reverse-td.pdf' % (img_root, game), bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    # plot_ppo()
    # plot_ddpg_td3()
    # plot_atari()

    # plot_reverse_td_lambda('auc')
    # plot_reverse_td_lambda('asym')
    # plot_reverse_td_robot_tasks_train()
    # plot_reverse_td_mutation_robot()
    plot_reverse_td_mutation_mujoco(0)
