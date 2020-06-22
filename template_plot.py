import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from deep_rl import *
import math


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


def plot_boyans_chain(game):
    log_root = './log/boyans-chain'
    # log_root = './tf_log/tf_log'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/gradient-dice/img'
    plotter = Plotter()
    games = [
        'BoyansChainTabular-v0',
        'BoyansChainLinear-v0',
    ]
    # game = games[0]
    # game = games[1]

    def score(y):
        print(y.shape)
        # assert y.shape[0] == 30
        return -np.mean(y[:, -1])
        # return -np.mean(y)

    gammas = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    # gammas = [0.1, 0.3]
    lrs = np.power(4.0, np.arange(-6, 0)).tolist()
    all_labels = []
    all_patterns = []

    activations = ['linear', 'squared', 'linear']

    for gamma in gammas:
        best = []
        algos = ['GradientDICE', 'GenDICE', 'DualDICE']
        for i, algo in enumerate(algos):
            ids = []
            labels = []
            patterns = []
            if gamma < 1:
                for lr in lrs:
                    template = '.*%s-activation_%s-algo_%s-discount_%s-lam_1-lr_%s-ridge_0-.*' % (
                        game, activations[i], algo, escape_float(gamma), escape_float(lr))
                    ids.append(lr)
                    label = r'%s($\alpha=4^{%s}$)' % (algo, round(math.log(lr, 4)))
                    labels.append(label)
                    patterns.append(template)
            else:
                for lr in lrs:
                    for ridge in [0, 0.001, 0.01, 0.1]:
                        template = '.*%s-activation_%s-algo_%s-discount_%s-lam_1-lr_%s-ridge_%s-.*' % (
                            game, activations[i], algo, escape_float(gamma), escape_float(lr), escape_float(ridge))
                        ids.append([lr, ridge])
                        label = r'%s($\alpha=4^{%s}, \xi=%s$)' % (algo, round(math.log(lr, 4)),
                                                                       ('10^{%s}' % round(math.log(ridge, 10)) if ridge else 0))
                        labels.append(label)
                        patterns.append(template)
            indices = plotter.select_best_parameters(
                patterns, score=score, root=log_root, tag='tau_loss', right_align=True, right_most=0)
            best.append([ids[indices[0]], labels[indices[0]], patterns[indices[0]]])

        all_labels.append([])
        all_patterns.append([])
        for _, label, pattern in best:
            all_patterns[-1].append(pattern)
            all_labels[-1].append(label)

    titles = [r'$\gamma = %s$' % (gamma) for gamma in gammas]

    fontsize = 25
    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        plt.figure(figsize=(3 * 5, 2 * 5))
        for i, title in enumerate(titles):
            plt.subplot(2, 3, i + 1)
            for j, p in enumerate(all_patterns[i]):
                label = all_labels[i][j]
                color = self.COLORS[j]
                marker = self.MARKERS[j]
                log_dirs = self.filter_log_dirs(pattern=p, **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se', marker=marker, markevery=10)
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std', marker=marker, markevery=10)
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color, marker=marker, markevery=10)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.xticks([0, 100], ['0', r'$3 \times 10^4$'], fontsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            if i in [0, 3]:
                plt.ylabel(r'MSE$(\tau)$', fontsize=fontsize)
            if i >= 3:
                plt.xlabel('Steps', fontsize=fontsize)
            y_min, _ = plt.gca().get_ylim()
            plt.gca().set_ylim(bottom=max(y_min, 0))
            plt.title(title, fontsize=fontsize)
            plt.legend(fontsize=15)

    plot_games(plotter,
               titles=titles,
               agg='mean_std',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='tau_loss',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/%s.pdf' % (img_root, game), bbox_inches='tight')


def plot_tmp():
    plotter = Plotter()
    games = [
        'BoyansChainTabular-v0',
        # 'BoyansChainLinear-v0',
    ]

    algo = 'GradientDICE'
    activation = 'linear'

    # algo = 'GenDICE'
    # activation = 'squared'
    gamma = 0.1
    patterns = []
    labels = []
    for lr in np.power(3, np.arange(0, 9)):
        template = '.*activation_%s-algo_%s-discount_%s-lam_1-lr_%s-ridge_0.*' % (
            activation, algo, escape_float(gamma), escape_float(lr))
        patterns.append(template)
        labels.append('%s' % lr)

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean_std',
                       downsample=0,
                       labels=labels,
                       right_align=True,
                       tag='tau_loss',
                       root='./tf_log/tf_log',
                       interpolation=0,
                       window=0,
                       right_most=0,
                       )

    plt.show()


def plot_mujoco():
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        # 'Reacher-v2',
    ]

    patterns = [
        'correction_GenDICE',
        'correction_GradientDICE',
        'correction_no',
    ]

    labels = patterns

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=True,
                       tag=plotter.RETURN_TEST,
                       root='./tf_log',
                       interpolation=0,
                       window=0,
                       right_most=0,
                       )

    plt.show()
    # plt.tight_layout()
    # plt.savefig('images/mujoco_eval.png', bbox_inches='tight')


def plot_mujoco_ope(game):
    log_root = './log/gradient-dice-mujoco-ope'
    # log_root = './tf_log/tf_log'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/gradient-dice/img'
    plotter = Plotter()
    games = [
        'Reacher-v2',
    ]

    def score(y):
        print(y.shape)
        # assert y.shape[0] == 30
        # return -np.mean(y[:, -5:])
        # return -np.mean(y[:, :50])
        return -np.mean(y[:, -1])
        # return -np.mean(y)

    gammas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    lams = [0.1, 1]
    lrs = [1e-2, 5e-3, 1e-3]
    # lrs = [1e-3, 1e-4, 1e-5]
    # lrs = [5e-4, 1e-4, 5e-5]
    # lrs = [1e-4, 5e-5, 1e-5]
    # lrs = [5e-5]

    all_labels = []
    all_patterns = []

    extra_labels = []
    extra_patterns = []
    for gamma in gammas:
        best = []
        algos = ['GradientDICE', 'GenDICE', 'DualDICE']
        # algos = ['GradientDICE', 'GenDICE']
        for i, algo in enumerate(algos):
            ids = []
            labels = []
            patterns = []
            if True:
                for lam in lams:
                    for lr in lrs:
                        template = '.*%s-correction_%s-discount_%s-lam_%s-lr_%s-.*' % (
                            game, algo, gamma, lam, lr
                        )
                        ids.append([lr])
                        # label = r'%s ($\alpha = 10^{%s}$)' % (algo, round(math.log(lr, 10)))
                        if algo in ['GradientDICE', 'GenDICE']:
                            label = r'%s ($\alpha=%s, \lambda=%s$)' % (algo, lr, lam)
                        else:
                            label = r'%s ($\alpha=%s$)' % (algo, lr)
                        labels.append(label)
                        patterns.append(template)

                    # if gamma in [0.1, 0.3] and algo == 'GenDICE' and lr == 1e-4:
                    #     extra_labels.append(label)
                    #     extra_patterns.append(template)
            indices = plotter.select_best_parameters(
                patterns, score=score, root=log_root, tag='perf_loss', right_align=True, right_most=0)
            best.append([ids[indices[0]], labels[indices[0]], patterns[indices[0]]])

        all_labels.append([])
        all_patterns.append([])
        for _, label, pattern in best:
            all_patterns[-1].append(pattern)
            all_labels[-1].append(label)

    # if lr_type == 'all':
    #     for i in range(1):
    #         all_labels[i].append(extra_labels[i])
    #         all_patterns[i].append(extra_patterns[i])

    titles = [r'$\gamma = %s$' % (gamma) for gamma in gammas]

    fontsize = 25
    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        plt.figure(figsize=(3 * 5, 2 * 5))
        for i, title in enumerate(titles):
            plt.subplot(2, 3, i + 1)
            for j, p in enumerate(all_patterns[i]):
                label = all_labels[i][j]
                color = self.COLORS[j]
                marker = self.MARKERS[j]
                log_dirs = self.filter_log_dirs(pattern=p, **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se', marker=marker, markevery=10)
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std', marker=marker, markevery=10)
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color, marker=marker, markevery=10)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            # plt.ylim([0, 0.1])
            # _, y_max = plt.gca().get_ylim()
            if i == 5:
                y_max = 0.05
            elif i == 4:
                y_max = 0.1
            else:
                y_max = 0.2
            plt.gca().set_ylim(bottom=0, top=y_max)
            plt.xticks([0, 100], ['0', r'$10^3$'], fontsize=fontsize)
            plt.title(title, fontsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            if i in [0, 3]:
                plt.ylabel(r'MSE($\rho$)', fontsize=fontsize)
            if i >= 3:
                plt.xlabel('Steps', fontsize=fontsize)
            plt.legend(fontsize=15)

    plot_games(plotter,
               titles=titles,
               agg='mean_std',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='perf_loss',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/%s.pdf' % (img_root, game), bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    # plot_boyans_chain('BoyansChainTabular-v0')
    # plot_boyans_chain('BoyansChainLinear-v0')
    # plot_mujoco_ope('Reacher-v2')

