import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
from deep_rl import *

IMG_DIR = '/Users/Shangtong/GoogleDrive/Paper/MVPI/img'


def plot_ppo():
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Swimmer-v2',
        'Hopper-v2',
        'Reacher-v2',
        'Ant-v2',
        'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    patterns = [
        '.*remark_ppo.*',
        # '.*meta_lr_0\.01-run.*',
        # '.*meta_lr_0\.001-run.*',
        # '.*meta_lr_0\.01-meta_prob_0\.1-run.*',
        # '.*meta_lr_0\.001-meta_prob_0\.1-run.*',
        '.*meta_lr_0\.01-meta_prob_1\.0-run.*',
        '.*meta_lr_0\.001-meta_prob_1\.0-run.*',
        # '.*lam_10-run.*',
        # '.*lam_1-run.*',
        # '.*lam_0\.1-run.*',
        # '.*lam_0\.01-run.*',
        # '.*lam_0\.001-run.*',
    ]

    patterns = [
        '.*lam_0\.1-run.*',
        # '.*lam_1-run.*',
        # '.*lam_10-run.*',
        '.*lr_0\.007-remark_mvp-run.*',
        '.*lr_7e-05-remark_mvp-run.*',
    ]

    patterns = [
        '.*b_10-lam_0.1-lr_0.0007-remark_tamar-run.*',
        '.*b_10-lam_0.1-lr_7e-05-remark_tamar-run.*',
        '.*b_10-lam_1-lr_0.0007-remark_tamar-run.*',
        '.*b_10-lam_10-lr_0.0007-remark_tamar-run.*',
        '.*b_50-lam_0.1-lr_0.0007-remark_tamar-run.*',
    ]

    patterns = [
        # '.*remark_tamar-run.*',
        # '.*remark_ppo-run.*',
        # '.*lam_0.1-remark_mvppo-run.*',
        # '.*lam_1-remark_mvppo-run.*',
        # '.*lam_10-remark_mvppo-run.*',
        # '.*meta_lr_0.001-meta_prob_0.1-run.*',
        # '.*meta_lr_0.001-meta_prob_0.5-run.*',
        # '.*meta_lr_0.001-meta_prob_1.0-run.*',
        # '.*remark_mvp-run.*',
        # '.*remark_tamar-run.*',
        # '.*remark_risk-run.*',
        '.*lam_0.1-remark_mva2c-run.*',
        '.*lam_1-remark_mva2c-run.*',
        '.*lam_10-remark_mva2c-run.*',
    ]

    # patterns = [
    #     '.*lam_10-lr_0\.0007-remark_risk-run.*',
    #     '.*lam_10-lr_7e-05-remark_risk-run.*',
    # ]

    # labels = [
    #     'PPO',
    #     'VarPPO'
    # ]

    labels = patterns

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       # agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       # root='./log/per-step-reward/meta-var-ppo',
                       # root='./log/per-step-reward/mvp-params',
                       # root='./log/per-step-reward/30runs',
                       root='./tf_log/tf_log',
                       interpolation=100,
                       window=10,
                       )

    plt.show()
    # plt.tight_layout()
    # plt.savefig('images/tmp.png', bbox_inches='tight')


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


def plot_risk_chain():
    plotter = Plotter()
    games = [
        'RiskChain-v0',
    ]

    num_samples = 1000

    patterns1 = [
        'lam_0-num_samples_%s-remark_off-policy-run' % (num_samples),
        'lam_1-num_samples_%s-remark_off-policy-run' % (num_samples),
        'lam_2-num_samples_%s-remark_off-policy-run' % (num_samples),
        'lam_4-num_samples_%s-remark_off-policy-run' % (num_samples),
        'lam_8-num_samples_%s-remark_off-policy-run' % (num_samples),
    ]

    patterns2 = [
        'lam_0-remark_off-policy-use_oracle_ratio_True-run',
        'lam_10-remark_off-policy-use_oracle_ratio_True-run',
    ]

    patterns = [patterns1, patterns2]

    labels = [
        r'$\lambda=0$',
        r'$\lambda=1$',
        r'$\lambda=2$',
        r'$\lambda=4$',
        r'$\lambda=8$',
    ]

    titles = ['Off-line MVPI']

    fontsize = 18

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        # plt.figure(figsize=(l * 5, 5))
        for i, title in enumerate(titles):
            # plt.subplot(1, l, i + 1)
            for j, p in enumerate(patterns[i]):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (games[0], p), **kwargs)
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
            plt.xlabel('Iterations', fontsize=fontsize)
            plt.xticks([0, 100], [0, 200], fontsize=fontsize)
            if i == 0:
                plt.ylabel(r'$\pi(a_0|s_0)$', horizontalalignment='right', fontsize=fontsize, rotation='horizontal')
            plt.yticks([0, 1], [0, 1], fontsize=fontsize)
            plt.ylim([0, 1])
            plt.title(title, fontsize=fontsize)
            plt.legend(fontsize=14)

    plot_games(plotter,
               games=games,
               patterns=patterns,
               # agg='mean_std',
               agg='mean',
               # agg='median',
               downsample=0,
               labels=labels,
               right_align=True,
               tag='pi_a0',
               root='./log/per-step-reward/risk-chain',
               # root='./tf_log/tf_log',
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/off-policy-mvpi.pdf' % (IMG_DIR), bbox_inches='tight')


def plot_mvpi_td3(lam=1):
    plotter = Plotter()
    games = [
        'InvertedPendulum-v2',
        'InvertedDoublePendulum-v2',
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Swimmer-v2',
        'Hopper-v2',
        'Reacher-v2',
        'Ant-v2',
    ]

    patterns = [
        '.*lam_0-remark_mvpi_td3-run.*',
        '.*lam_%s-remark_mvpi_td3-run.*' % (lam),
        '.*lam_%s-remark_trvo-run.*' % (lam),
        '.*lam_%s-remark_mvp-run.*' % (lam),
        '.*lam_%s-remark_risk-run.*' % (lam),
        '.*lam_%s-remark_tamar-run.*' % (lam),
    ]

    labels = [
        'TD3',
        'MVPI-TD3',
        'TRVO',
        'MVP',
        'Prashanth',
        'Tamar',
    ]

    # labels = patterns

    fontsize = 20

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        plt.figure(figsize=(4 * 5, 2 * 4.5))
        for i, game in enumerate(games):
            plt.subplot(2, 4, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, p), **kwargs)
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
            if i >= 4:
                plt.xlabel('Steps', fontsize=fontsize)
            plt.xticks([0, 1e6], ['0', r'$10^6$'], fontsize=fontsize)
            if i % 4 == 0:
                plt.ylabel('Episode Return', fontsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.title(game, fontsize=fontsize)
            if i == 0:
                plt.legend(fontsize=fontsize)

    plot_games(plotter,
               games=games,
               patterns=patterns,
               # agg='mean_std',
               agg='mean',
               downsample=0,
               labels=labels,
               right_align=False,
               tag=plotter.RETURN_TEST,
               root='./log/per-step-reward/10runs',
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/mvpi-td3-%s.pdf' % (IMG_DIR, lam), bbox_inches='tight')


def plot_mvpi_td3_mean_per_step_reward(lam=1):
    plotter = Plotter()
    games = [
        'InvertedPendulum-v2',
        'InvertedDoublePendulum-v2',
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Swimmer-v2',
        'Hopper-v2',
        'Reacher-v2',
        'Ant-v2',
    ]

    patterns = [
        '.*lam_%s-remark_mvpi_td3-run.*' % (lam),
        '.*lam_%s-remark_trvo-run.*' % (lam),
    ]

    labels = [
        'MVPI-TD3',
        'TRVO',
    ]

    # labels = patterns

    fontsize = 20

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        plt.figure(figsize=(4 * 5, 2 * 4.5))
        for i, game in enumerate(games):
            plt.subplot(2, 4, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                tag_prefix = kwargs['tag']
                ys = []
                for tag in ['mean', 'std']:
                    kwargs['tag'] = '%s_%s' % (tag_prefix, tag)
                    log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, p), **kwargs)
                    x, y = self.load_results(log_dirs, **kwargs)
                    ys.append(y)
                kwargs['tag'] = tag_prefix
                y_mean, y_std = ys
                y = y_mean - lam * np.power(y_std, 2)
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
            if i >= 4:
                plt.xlabel('Steps', fontsize=fontsize)
            plt.xticks([0, 1e6], ['0', r'$10^6$'], fontsize=fontsize)
            if i % 4 == 0:
                plt.ylabel(r'$J_{\text{reward}}$', fontsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.title(game, fontsize=fontsize)
            if i == 0:
                plt.legend(fontsize=fontsize)

    plot_games(plotter,
               games=games,
               patterns=patterns,
               # agg='mean_std',
               agg='mean',
               downsample=0,
               labels=labels,
               right_align=False,
               tag='per_step_reward_test',
               root='./log/per-step-reward/10runs',
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/mvpi-td3-per-step-reward-%s.pdf' % (IMG_DIR, lam), bbox_inches='tight')


def generate_table(lam=1, reload=False):
    plotter = Plotter()
    games = [
        'InvertedPendulum-v2',
        'InvertedDoublePendulum-v2',
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Swimmer-v2',
        'Hopper-v2',
        'Reacher-v2',
        'Ant-v2',
        # 'Humanoid-v2',
    ]

    patterns = [
        '.*lam_0-remark_mvpi_td3-run.*',
        '.*lam_%s-remark_trvo-run.*' % (lam),
        '.*lam_%s-remark_mvpi_td3-run.*' % (lam),
        # '.*lam_%s-remark_mvp-run.*' % (lam),
        # '.*lam_%s-remark_tamar-run.*' % (lam),
        # '.*lam_%s-remark_risk-run.*' % (lam),
    ]

    labels = [
        'TD3',
        'TRVO',
        'MVPI-TD3',
    ]

    kwargs = dict(games=games,
                  patterns=patterns,
                  agg='mean',
                  downsample=0,
                  right_align=True,
                  tag='EOT_eval',
                  root='./log/per-step-reward/10runs',
                  interpolation=0,
                  window=0)

    stats = dict()

    if reload:
        for i, game in enumerate(games):
            for j, p in enumerate(patterns):
                log_dirs = plotter.filter_log_dirs(pattern='.*%s.*%s' % (game, p), **kwargs)
                x, y = plotter.load_results(log_dirs, **kwargs)
                print(y.shape)
                mean = np.mean(y, axis=1)
                var = np.std(y, axis=1) ** 2 + 1e-5
                stats[(i, j)] = [mean, var]

        with open('data/per-step-reward/eval_perf-%s.bin' % (lam), 'wb') as f:
            pickle.dump(stats, f)

    with open('data/per-step-reward/eval_perf-%s.bin' % (lam), 'rb') as f:
        stats = pickle.load(f)

    # strs = []
    # for j in range(1, len(patterns)):
    #     str = '%s' % (labels[j])
    #     for i, game in enumerate(games):
    #         mean_baseline, var_baseline = stats[(i, 0)]
    #         mean_algo, var_algo = stats[(i, j)]
    #         mean_baseline = np.mean(mean_baseline)
    #         var_baseline = np.mean(var_baseline)
    #         mean_algo = np.mean(mean_algo)
    #         var_algo = np.mean(var_algo)
    #         J_baseline = mean_baseline - lam * var_baseline
    #         J_algo = mean_algo - lam * var_algo
    #         perf_improv = np.round((J_algo - J_baseline) / np.abs(J_baseline) * 100).astype(np.int)
    #         str = str + '& %s\\%%' % (perf_improv)
    #     str = str + '\\\\ \\hline'
    #     strs.append(str)
    # for str in strs:
    #     print(str)

    strs = []
    for i, game in enumerate(games):
        str = '%s' % (games[i][:-3])
        for j in range(1, len(patterns)):
            mean_baseline, var_baseline = stats[(i, 0)]
            mean_algo, var_algo = stats[(i, j)]
            mean_baseline = np.mean(mean_baseline)
            var_baseline = np.mean(var_baseline) + 1e-2
            mean_algo = np.mean(mean_algo)
            var_algo = np.mean(var_algo) + 1e-2
            print(games[i], labels[0], mean_baseline, var_baseline)
            print(games[i], labels[j], mean_algo, var_algo)
            J_baseline = mean_baseline - lam * var_baseline
            J_algo = mean_algo - lam * var_algo
            sr_baseline = mean_baseline / (var_baseline ** 0.5)
            sr_algo = mean_algo / (var_algo ** 0.5)
            perf_improv = np.round((J_algo - J_baseline) / np.abs(J_baseline) * 100).astype(np.int)
            mean_improv = np.round((mean_algo - mean_baseline)/ np.abs(mean_baseline) * 100).astype(np.int)
            var_improv = np.round((var_algo - var_baseline)/ np.abs(var_baseline) * 100).astype(np.int)
            sr_improv = np.round((sr_algo - sr_baseline)/ np.abs(sr_baseline) * 100).astype(np.int)
            str = str + '& %s\\%% & %s\\%% & %s\\%% & %s\\%%' % (perf_improv, mean_improv, var_improv, sr_improv)
        str = str + '\\\\ \\hline'
        strs.append(str)
    for str in strs:
        print(str)


if __name__ == '__main__':
    mkdir('images')
    # plot_ppo()
    # plot_ddpg_td3()
    # plot_atari()

    # plot_mvpi_td3(0.5)
    # plot_mvpi_td3(1)
    # plot_mvpi_td3(2)
    # generate_table(0.5)
    # generate_table(1)
    # generate_table(2)
    # plot_mvpi_td3_mean_per_step_reward(0.5)
    # plot_mvpi_td3_mean_per_step_reward(1)
    # plot_mvpi_td3_mean_per_step_reward(2)
    # plot_risk_chain()
