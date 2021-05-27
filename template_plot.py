import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
import os

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
from deep_rl import *
import math
from concurrent import futures


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


def get_best_hp_boyans_chain(pi0):
    patterns = [
        'algo_GradientDICE.*hp_0-',
        'algo_FQE.*hp_0-',
        'algo_GQ1.*hp_0-',
        'algo_GQ2.*hp_0-',
    ]
    all_ids = [
        np.arange(240),
        np.arange(60),
        np.arange(60),
        np.arange(60),
    ]
    def score(y):
        print(y.shape)
        try:
            score = -np.mean(y[:, -1])
        except Exception as e:
            score = -float('inf')
        return score
    log_root = './log/DifferentialGQ/boyan'
    plotter = Plotter()
    best_hp = {pi0: {}}
    for mu0 in [np.round(1 - pi0, 1), pi0, 0.5]:
        cur_patterns = ['%s.*mu0_%s.*pi0_%s-' % (p, mu0, pi0) for p in patterns]
        info = plotter.reduce_patterns(cur_patterns, log_root, 'rate_loss', all_ids, score)
        best_hp[pi0][mu0] = info
    return best_hp


def plot_boyans_chain(reload=False):
    # log_root = './tf_log/tf_log'
    log_root = './log/DifferentialGQ/boyan'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/ope-differential/img'
    plotter = Plotter()
    game = 'BoyansChainLinear-v0',

    labels = [
        'GradientDICE',
        'Diff-SGQ',
        'Diff-GQ1',
        'Diff-GQ2',
    ]
    target_policies = [0.1, 0.3, 0.5, 0.7, 0.9]
    # target_policies = [0.1, 0.3, 0.5, 0.7, 0.9]
    titles = [str(p) for p in target_policies]

    if reload:
        boyan_chain_best_hp = {}
        with futures.ProcessPoolExecutor() as pool:
            for best_hp in pool.map(get_best_hp_boyans_chain, target_policies):
                boyan_chain_best_hp.update(best_hp)
        with open('./data/DifferentialGQ/boyan_chain_best_hp.pkl', 'wb') as f:
            pickle.dump(boyan_chain_best_hp, f)
    else:
        with open('./data/DifferentialGQ/boyan_chain_best_hp.pkl', 'rb') as f:
            boyan_chain_best_hp = pickle.load(f)

    fontsize = 22

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        n_row = 3
        plt.figure(figsize=(l * 5, n_row * 5))
        for k in range(n_row):
            for i, pi0 in enumerate(target_policies):
                mu0 = [pi0, 0.5, np.round(1 - pi0, 1)][k]
                info = boyan_chain_best_hp[pi0][mu0]
                fid = k * len(target_policies) + i + 1
                if fid == 8 or fid == 13:
                    continue
                plt.subplot(n_row, l, fid)
                for j, p in enumerate(info['patterns']):
                    # label = '%s (%s)' % (labels[j], info['ids'][j])
                    label = labels[j]
                    color = self.COLORS[j]
                    log_dirs = self.filter_log_dirs(pattern='.*%s.*' % (p), **kwargs)
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
                plt.xticks([0, 100], ['0', r'$5 \times 10^3$'], fontsize=fontsize)
                if fid >= 11:
                    plt.xlabel('Steps', fontsize=fontsize)
                plt.tick_params(axis='y', labelsize=fontsize)
                if fid % 5 == 1:
                    # plt.ylabel(r'$|\hat{r} - \bar{r}_\pi|$', rotation='horizontal', fontsize=10)
                    plt.ylabel(r'$|\bar{\hat{r}} - \bar{r}_\pi|$', rotation='horizontal', fontsize=fontsize, labelpad=35)
                # if not i:
                # plt.ylabel(r'MSE$(\tau)$', fontsize=fontsize)
                # y_min, _ = plt.gca().get_ylim()
                # plt.gca().set_ylim(bottom=max(y_min, 0))
                plt.title(r'$\pi_0=%s, \mu_0=%s$' % (pi0, mu0), fontsize=fontsize)
                if fid == 3:
                    plt.legend(fontsize=fontsize, bbox_to_anchor=(1, -1))

    plot_games(plotter,
               titles=titles,
               agg='mean_std',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='rate_loss',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    # plt.tight_layout()
    # plt.savefig('%s/boyans_chain.pdf' % (img_root), bbox_inches='tight')
    plt.savefig('%s/boyans_chain.pdf' % (img_root))


MUJOCO_LOSS_TYPE = 'online_rate_loss'
def get_best_hp_mujoco(game):
    patterns = [
        'algo_GradientDICE.*hp_0-',
        'algo_FQE-target.*hp_0-',
        'algo_GQ1.*hp_0-',
        'algo_GQ2.*hp_0-',
    ]
    all_ids = [
        np.arange(15),
        np.arange(5),
        np.arange(5),
        np.arange(5),
    ]
    def score(y):
        print(y.shape)
        try:
            score = -np.mean(y[:, -1])
        except Exception as e:
            score = -float('inf')
        return score
    log_root = './log/DifferentialGQ/mujoco'
    plotter = Plotter()
    best_hp = {game: {}}
    for noise in [0.1, 0.5, 0.9]:
        cur_patterns = ['%s.*%s.*noise_%s-' % (game, p, noise) for p in patterns]
        info = plotter.reduce_patterns(cur_patterns, log_root, MUJOCO_LOSS_TYPE, all_ids, score)
        best_hp[game][noise] = info
    return best_hp


def plot_mujoco(group, reload=False, loss='online_rate_loss'):
    log_root = './log/DifferentialGQ/mujoco'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/ope-differential/img'
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
    ]
    global MUJOCO_LOSS_TYPE
    MUJOCO_LOSS_TYPE = loss

    labels = [
        'GradientDICE',
        'Diff-SGQ',
        'Diff-GQ1',
        'Diff-GQ2',
    ]
    titles = games

    if reload:
        mujoco_best_hp = {}
        with futures.ProcessPoolExecutor() as pool:
            for best_hp in pool.map(get_best_hp_mujoco, games):
                mujoco_best_hp.update(best_hp)
        with open('./data/DifferentialGQ/mujoco_best_hp_%s.pkl' % (MUJOCO_LOSS_TYPE), 'wb') as f:
            pickle.dump(mujoco_best_hp, f)
    else:
        with open('./data/DifferentialGQ/mujoco_best_hp_%s.pkl' % (MUJOCO_LOSS_TYPE), 'rb') as f:
            mujoco_best_hp = pickle.load(f)

    fontsize = 22

    noise = [[0.1, 0.5], [0.9]][group]
    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        n_row = len(noise)
        plt.figure(figsize=(l * 5, n_row * 5))
        for k in range(n_row):
            for i, game in enumerate(games):
                info = mujoco_best_hp[game][noise[k]]
                fid = k * len(games) + i + 1
                plt.subplot(n_row, l, fid)
                for j, p in enumerate(info['patterns']):
                    # label = '%s (%s)' % (labels[j], info['ids'][j])
                    label = labels[j]
                    color = self.COLORS[j]
                    log_dirs = self.filter_log_dirs(pattern='.*%s.*' % (p), **kwargs)
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
                plt.xticks([0, 100], ['0', r'$10^3$'], fontsize=fontsize)
                if k == n_row - 1:
                    plt.xlabel('Steps', fontsize=fontsize)
                plt.tick_params(axis='y', labelsize=fontsize)
                if fid % 4 == 1:
                    plt.ylabel(r'$|\bar{\hat{r}} - \bar{r}_\pi|$', rotation='horizontal', fontsize=fontsize, labelpad=35)
                plt.title(r'%s ($\sigma=%s$)' % (game[:-3], noise[k]), fontsize=fontsize)
                if not k and not i:
                    plt.legend(fontsize=fontsize)

    plot_games(plotter,
               titles=titles,
               agg='mean_std',
               downsample=0,
               right_align=True,
               right_most=0,
               tag=MUJOCO_LOSS_TYPE,
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/mujoco_group_%d.pdf' % (img_root, group), bbox_inches='tight')


def plot_tmp():
    # log_root = './log/boyans-chain'
    # log_root = './tf_log/tf_log'
    log_root = './log/DifferentialGQ/boyan'
    # img_root = '/Users/Shangtong/GoogleDrive/Paper/gradient-dice/img'
    plotter = Plotter()
    game = 'BoyansChainLinear-v0',

    # game = games[0]
    # game = games[1]

    def score(y):
        print(y.shape)
        return -np.mean(y[:, -1])

    patterns = [
        # 'algo_GenDICE.*hp_0-',
        # 'algo_GradientDICE.*hp_0-',
        # 'algo_FQE.*hp_0-',
        'algo_GQ1.*hp_14-',
        'algo_GQ1.*hp_8-',
        'algo_GQ1.*hp_11-',
        'algo_GQ1.*hp_17-',
        # 'algo_GQ2.*hp_0-',
    ]
    labels = [
        # 'GenDICE',
        'GradientDICE',
        'GradientDICE',
        'GradientDICE',
        'GradientDICE',
        # 'Differential FQE',
        # 'Differential GQ1',
        # 'Differential GQ2',
    ]
    # target_policies = [0.1, 0.3, 0.5, 0.7, 0.9]
    target_policies = [0.9]
    titles = [str(p) for p in target_policies]

    fontsize = 22

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(titles)
        plt.figure(figsize=(l * 5, 5))
        for i, title in enumerate(titles):
            pi0 = target_policies[i]
            mu0 = np.round(1 - pi0, 1)
            # mu0 = pi0
            # mu0 = 0.5
            cur_patterns = ['%s.*mu0_%s.*pi0_%s-' % (p, mu0, pi0) for p in patterns]
            # info = plotter.reduce_patterns(cur_patterns, log_root, 'rate_loss', np.arange(30), score)
            info = dict(patterns=cur_patterns)
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(info['patterns']):
                label = '%s' % (labels[j])
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*' % (p), **kwargs)
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
            plt.xticks([0, 100], ['0', r'$5 \times 10^3$'], fontsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            # if not i:
            # plt.ylabel(r'MSE$(\tau)$', rotation='horizontal', fontsize=10)
            # plt.ylabel(r'MSE$(\tau)$', fontsize=fontsize)
            # y_min, _ = plt.gca().get_ylim()
            # plt.gca().set_ylim(bottom=max(y_min, 0))
            plt.title(title, fontsize=fontsize)
            plt.legend(fontsize=12)

    plot_games(plotter,
               titles=titles,
               agg='mean_std',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='rate_loss',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    # plt.tight_layout()
    # plt.savefig('%s/%s.pdf' % (img_root, game), bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    # plot_boyans_chain(False)
    # plot_mujoco(0, False)
    plot_mujoco(1, False)

    # plot_tmp()
