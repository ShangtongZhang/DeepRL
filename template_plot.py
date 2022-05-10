import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent import futures
import math
from numpy.lib.utils import info
import pandas as pd

from numpy.core.fromnumeric import var

plt.rc('text', usetex=True)
import os

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
from deep_rl import *

img_root = '/Users/Shangtong/GoogleDrive/Paper/target-network/img'
log_folder = '/Volumes/Data/DeepRL'
fontsize = 22


def get_best_hp_baird(info):
    id, (patterns, all_ids) = info
    def score(y):
        print(y.shape)
        try:
            score = -np.mean(y[:, -1])
        except Exception as e:
            score = -float('inf')
        return score
    log_root = './log/vretd'
    plotter = Plotter()
    info = plotter.reduce_patterns(patterns, log_root, 'error', all_ids, score)
    return {id: info}


def get_best_hp_cartpole(info):
    id, (patterns, all_ids) = info
    def score(y):
        print(y.shape)
        try:
            score = np.mean(y[:, -1])
            # score = np.mean(y)
        except Exception as e:
            score = -float('inf')
        if y.shape[1] < 90:
            score = -float('inf')
        return score
    log_root = f'{log_folder}/log/vretd/cartpole1000'
    plotter = Plotter()
    info = plotter.reduce_patterns(patterns, log_root, 'episodic_return_test', all_ids, score)
    return {id: info}


def plot_baird(group, reload=False):
    log_root = f'{log_folder}/log/vretd/baird'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/truncated_etd/img'
    plotter = Plotter()

    if group == 0 or group == 3:
        game = 'BairdPrediction-v0',
        pi_dashed = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
        patterns = []
        hps = []
        for pi in pi_dashed:
            patterns.append([ 
                'hard.*BairdPrediction.*hp_0-.*n_-1-.*pi_dashed_%s-' % (pi),
                'hard.*BairdPrediction.*hp_0-.*n_0-.*pi_dashed_%s-' % (pi),
                'hard.*BairdPrediction.*hp_0-.*n_2-.*pi_dashed_%s-' % (pi),
                'hard.*BairdPrediction.*hp_0-.*n_4-.*pi_dashed_%s-' % (pi),
                'hard.*BairdPrediction.*hp_0-.*n_8-.*pi_dashed_%s-' % (pi),
                'soft.*BairdPrediction.*hp_0-.*n_-1-.*pi_dashed_%s-' % (pi),
            ])
            hps.append([np.arange(20)] * 5 + [np.arange(80)])
        titles = [r"$\pi({dashed}|s) = %s$" % (pi) for pi in pi_dashed]
    elif group == 1 or group == 2:
        game = 'BairdControl-v0',
        tau = [0.01, 0.1, 1]
        titles = [r"$\tau= %s$" % (t) for t in tau]
        patterns = []
        hps = []
        for t in tau:
            if group == 1:
                flag = "False"
            else:
                flag = "True"
            patterns.append([
                'hard.*BairdControl.*hp_0-.*n_-1-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'hard.*BairdControl.*hp_0-.*n_0-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'hard.*BairdControl.*hp_0-.*n_2-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'hard.*BairdControl.*hp_0-.*n_4-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'hard.*BairdControl.*hp_0-.*n_8-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'soft.*BairdControl.*hp_0-.*n_-1-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
            ])
            hps.append([np.arange(20)] * 5 + [np.arange(80)])
    else:
        raise NotImplementedError

    labels = [
        'n=\infty',
        'n=0',
        'n=2',
        'n=4',
        'n=8',
        'n=16',
        'n=32',
    ]

    filename = './data/VRETD/baird_%s.pkl' % (group % 3)
    if reload:
        patterns_with_best_hp = {}
        with futures.ProcessPoolExecutor() as pool:
            for best_hp in pool.map(get_best_hp_baird, enumerate(zip(patterns, hps))):
                patterns_with_best_hp.update(best_hp)
        patterns_with_best_hp = [patterns_with_best_hp[i] for i in range(len(patterns))]
        with open(filename, 'wb') as f:
            pickle.dump(patterns_with_best_hp, f)
    else:
        with open(filename, 'rb') as f:
            patterns_with_best_hp = pickle.load(f)

    fontsize = 22
    legend_size = 12

    hard_hps = HyperParameters(OrderedDict(lr=0.1 * np.power(2.0, -np.arange(20))))
    soft_hps = HyperParameters(OrderedDict(
        lr=0.1 * np.power(2.0, -np.arange(20)),
        beta=[0.1, 0.2, 0.4, 0.8]))

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        if group % 3 == 0:
            n_row = 2
            n_col = len(titles) // 2
        else:
            n_row = 1
            n_col = len(titles)
        plt.figure(figsize=(n_col * 5.2, n_row * 5))
        for k in range(n_row):
            for i in range(n_col):
                ind = n_col * k + i
                info = patterns_with_best_hp[ind]
                plt.subplot(n_row, n_col, ind + 1)
                for j, p in enumerate(info['patterns']):
                    if group == 3:
                        p = f'{p}run-1-'
                    if 'hard' in p:
                        best_hp = hard_hps[info['ids'][j]]
                        # if group == 3:
                            # label = r'$%s$' % (labels[j])
                        # else:
                        label = r'$%s\,(\alpha = 0.1 \times 2^{-%s})$' % (labels[j], best_hp)
                    elif 'soft' in p:
                        best_hp = soft_hps[info['ids'][j]]
                        best_lr = best_hp.param['lr']
                        best_beta = best_hp.param['beta']
                        best_lr = int(np.round(math.log(best_lr * 10, 2)))
                        # if group == 3:
                            # label = r'$\beta=%s$' % (best_beta)
                        # else:
                        label = r'$\beta=%s \, (\alpha = 0.1 \times 2^{%s})$' % (best_beta, best_lr)
                    else:
                        raise NotImplementedError
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
                        raise NotImplementedError
                plt.xticks([0, 100], ['0', r'$5 \times 10^5$'], fontsize=fontsize)
                plt.ylim([0, 30])
                plt.title(titles[ind], fontsize=fontsize)
                plt.tick_params(axis='y', labelsize=fontsize)
                if group == 0:
                    if i % 3 == 0:
                        plt.ylabel(r'$||Xw_t - v_\pi||$', rotation='horizontal', fontsize=fontsize, labelpad=55)
                    if k == 1:
                        plt.xlabel('Steps', fontsize=fontsize)
                else:
                    if i % 3 == 0:
                        plt.ylabel(r'$||Xw_t - q_*||$', rotation='horizontal', fontsize=fontsize, labelpad=55)
                    plt.xlabel('Steps', fontsize=fontsize)
                # if fid % 5 == 1:
                    # plt.ylabel(r'$|\hat{r} - \bar{r}_\pi|$', rotation='horizontal', fontsize=10)
                # if not i:
                # plt.ylabel(r'MSE$(\tau)$', fontsize=fontsize)
                # y_min, _ = plt.gca().get_ylim()
                # plt.gca().set_ylim(bottom=max(y_min, 0))
                # if fid == 3:
                # plt.legend(fontsize=fontsize, bbox_to_anchor=(1, -1))
                plt.legend(fontsize=legend_size, loc="upper right")

    plot_games(plotter,
               titles=titles,
               agg='mean',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='error',
               root=log_root,
               interpolation=0,
               window=10 if group == 3 else 0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/baird_group_%s.pdf' % (img_root, group), bbox_inches='tight')


def read_variance_impl(group, reload=False):
    variance_info = {}
    log_root = './log/vretd'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/truncated_etd/img'
    plotter = Plotter()

    if group == 0:
        game = 'BairdPrediction-v0',
        pi_dashed = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
        patterns = []
        hps = []
        for pi in pi_dashed:
            patterns.append([ 
                'hard.*BairdPrediction.*hp_0-.*n_-1-.*pi_dashed_%s-' % (pi),
                'hard.*BairdPrediction.*hp_0-.*n_0-.*pi_dashed_%s-' % (pi),
                'hard.*BairdPrediction.*hp_0-.*n_2-.*pi_dashed_%s-' % (pi),
                'hard.*BairdPrediction.*hp_0-.*n_4-.*pi_dashed_%s-' % (pi),
                'hard.*BairdPrediction.*hp_0-.*n_8-.*pi_dashed_%s-' % (pi),
                'soft.*BairdPrediction.*hp_0-.*n_-1-.*pi_dashed_%s-' % (pi),
            ])
            hps.append([np.arange(20)] * 5 + [np.arange(80)])
        titles = [r"$\pi({dashed}|s) = %s$" % (pi) for pi in pi_dashed]
    elif group == 1 or group == 2:
        game = 'BairdControl-v0',
        tau = [0.01, 0.1, 1]
        titles = [r"$\tau= %s$" % (t) for t in tau]
        patterns = []
        hps = []
        for t in tau:
            if group == 1:
                flag = "False"
            else:
                flag = "True"
            patterns.append([
                'hard.*BairdControl.*hp_0-.*n_-1-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'hard.*BairdControl.*hp_0-.*n_0-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'hard.*BairdControl.*hp_0-.*n_2-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'hard.*BairdControl.*hp_0-.*n_4-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'hard.*BairdControl.*hp_0-.*n_8-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
                'soft.*BairdControl.*hp_0-.*n_-1-.*softmax_mu_%s-.*tau_%s-' % (flag, tau),
            ])
            hps.append([np.arange(20)] * 5 + [np.arange(80)])
    else:
        raise NotImplementedError

    labels = [
        'n=\infty',
        'n=0',
        'n=2',
        'n=4',
        'n=8',
        'n=16',
        'n=32',
    ]

    filename = './data/VRETD/baird_%s.pkl' % (group)
    if reload:
        patterns_with_best_hp = {}
        with futures.ProcessPoolExecutor() as pool:
            for best_hp in pool.map(get_best_hp_baird, enumerate(zip(patterns, hps))):
                patterns_with_best_hp.update(best_hp)
        patterns_with_best_hp = [patterns_with_best_hp[i] for i in range(len(patterns))]
        with open(filename, 'wb') as f:
            pickle.dump(patterns_with_best_hp, f)
    else:
        with open(filename, 'rb') as f:
            patterns_with_best_hp = pickle.load(f)

    fontsize = 22
    legend_size = 12

    hard_hps = HyperParameters(OrderedDict(lr=0.1 * np.power(2.0, -np.arange(20))))
    soft_hps = HyperParameters(OrderedDict(
        lr=0.1 * np.power(2.0, -np.arange(20)),
        beta=[0.1, 0.2, 0.4, 0.8]))

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        if group == 0:
            n_row = 2
            n_col = len(titles) // 2
        else:
            n_row = 1
            n_col = len(titles)
        plt.figure(figsize=(n_col * 5.2, n_row * 5))
        for k in range(n_row):
            for i in range(n_col):
                ind = n_col * k + i
                info = patterns_with_best_hp[ind]
                plt.subplot(n_row, n_col, ind + 1)
                for j, p in enumerate(info['patterns']):
                    if 'hard' in p:
                        best_hp = hard_hps[info['ids'][j]]
                        label = r'$%s\,(\alpha = 0.1 \times 2^{-%s})$' % (labels[j], best_hp)
                    elif 'soft' in p:
                        best_hp = soft_hps[info['ids'][j]]
                        best_lr = best_hp.param['lr']
                        best_beta = best_hp.param['beta']
                        best_lr = int(np.round(math.log(best_lr * 10, 2)))
                        label = r'$\beta=%s \, (\alpha = 0.1 \times 2^{%s})$' % (best_beta, best_lr)
                    else:
                        raise NotImplementedError
                    color = self.COLORS[j]
                    log_dirs = self.filter_log_dirs(pattern='.*%s.*' % (p), **kwargs)
                    x, y = self.load_results(log_dirs, **kwargs)
                    variance_info[(k, i, j)] = {
                        'last_var': np.std(y[:, -1]) ** 2,
                        'total_var': np.sum([np.std(y[:, t]) ** 2 for t in np.arange(y.shape[1])]),
                        'last_mean': np.mean(y[:, -1])}

    plot_games(plotter,
               titles=titles,
               agg='mean',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='error',
               root=log_root,
               interpolation=0,
               window=0,
               )
    return variance_info


def read_variance(group, reload=False):
    filename = './data/VRETD/baird_variance.pkl'
    if reload:
        variance_info = {}
        for group in [0, 1, 2]:
            variance_info[group] = read_variance_impl(group)
        with open(filename, 'wb') as f:
            pickle.dump(variance_info, f)
    else:
        with open(filename, 'rb') as f:
            variance_info = pickle.load(f)
    data = []
    mask = []
    if group ==0 :
        n_rows = 2
        rows = ['$\pi(dashed=%s|s)$' % (pi) for pi in [0, 0.02, 0.04, 0.06, 0.08, 0.1]]
    else:
        n_rows = 1
        rows = ['$\\tau=%s$' % (tau) for tau in [0, 0.01, 0.1]]
    n_cols = 3
    for row in range(n_rows):
        for col in range(n_cols):
            data.append([variance_info[group][(row, col, p)]['total_var'] for p in range(6)])
            mask.append([variance_info[group][(row, col, p)]['last_mean'] for p in range(6)])
    data = np.array(data)
    data = data / 101
    print(data)
    data = np.round(np.log10(data)).astype(np.int)
    mask = np.array(mask)
    mask = (mask < 5)
    best_j = np.argmin(np.where(mask, data, float('inf')), axis=-1)
    latex_data = np.empty(data.shape, dtype=np.object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                latex_data[i, j] = '$10^{%s}$' % (data[i, j])
                if j == best_j[i]:
                    latex_data[i, j] = '\\textbf{%s}' % latex_data[i, j]
            else:
                latex_data[i, j] = '-'
    columns = ['$n=\\infty$', '$n=0$', '$n=2$', '$n=4$', '$n=8$', '$\\beta=0.8$']
    data = pd.DataFrame(
        data=latex_data, 
        columns=columns)
    data.index = rows
    print(data.to_latex(escape=False,
        column_format='c|cccccc'))


def plot_cartpole(group, reload=False):
    log_root = f'{log_folder}/log/vretd/cartpole1000'
    img_root = '/Users/Shangtong/GoogleDrive/Paper/truncated_etd/img'
    plotter = Plotter()

    game = 'CartPole-v0',
    colors = None
    if group == 0:
        ns = [-1, 0, 2, 4, 8]
        epses = [0.95]
        patterns = [[
            f'beta_0.99.*eps_{eps}-hp_0-.*n_{n}.*-tau_0.01.*-run' for n in ns
        ] for eps in epses]
        hps = [[np.arange(20)] * len(ns) for eps in epses]
        titles = [f'{eps}' for eps in epses]
        labels = [
            'n=\infty',
            'n=0',
            'n=2',
            'n=4',
            'n=8',
        ]
    elif group == 1:
        epses = [0.95]
        betas = [0.2, 0.99]
        patterns = [[
            f'beta_{beta}.*eps_{eps}-hp_0-.*n_-1.*-tau_0.01.*-run' for beta in betas
        ] + [
            f'beta_0.99.*eps_{eps}-hp_0-.*n_0.*-tau_0.01.*-run',
            f'beta_0.99.*eps_{eps}-hp_0-.*n_4.*-tau_0.01.*-run',
        ] for eps in epses]
        hps = [[np.arange(20)] * (len(betas) + 2) for eps in epses]
        labels = [
            '\\beta=0.2',
            'n=\infty',
            'n=0',
            'n=4',
        ]
        titles = ['x']
        colors = ['purple', 'blue', 'green', 'black']
    else:
        raise NotImplementedError


    # labels = ['Emphatic SARSA', 'Truncated Emphatic SARSA']
    # labels = betas

    # get_best_hp_baird((0, (patterns[0], hps[0])))
    filename = './data/VRETD/cartpole_%s.pkl' % (group)
    if reload:
        patterns_with_best_hp = {}
        with futures.ProcessPoolExecutor() as pool:
            for best_hp in pool.map(get_best_hp_cartpole, enumerate(zip(patterns, hps))):
                patterns_with_best_hp.update(best_hp)
        patterns_with_best_hp = [patterns_with_best_hp[i] for i in range(len(patterns))]
        with open(filename, 'wb') as f:
            pickle.dump(patterns_with_best_hp, f)
    else:
        with open(filename, 'rb') as f:
            patterns_with_best_hp = pickle.load(f)

    fontsize = 22
    legend_size = 12

    # patterns_with_best_hp = patterns_with_best_hp[9:]

    lr_hps = HyperParameters(OrderedDict(lr=0.1 * np.power(2.0, -np.arange(20))))

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        if group == 0:
            n_row = 1 
            n_col = len(epses) 
        else:
            n_row = 1
            n_col = len(epses)
        plt.figure(figsize=(n_col * 5.5, n_row * 5))
        for k in range(n_row):
            for i in range(n_col):
                ind = n_col * k + i
                info = patterns_with_best_hp[ind]
                # info = dict(
                    # patterns=[info['patterns'][0], info['patterns'][-1]],
                    # ids=[info['ids'][0], info['ids'][-1]],
                # )
                plt.subplot(n_row, n_col, ind + 1)
                for j, p in enumerate(info['patterns']):
                    # p = f'{p}-0'
                    if group == 0:
                        best_hp = lr_hps[info['ids'][j]]
                        label = r'$%s\,(\alpha = 0.1 \times 2^{-%s})$' % (labels[j], best_hp)
                    elif group == 3:
                        best_hp = lr_hps[info['ids'][j]]
                        label = r'$%s\,(\alpha = 0.1 \times 2^{-%s})$' % (labels[j], best_hp)
                    else:
                        raise NotImplementedError
                    color = self.COLORS[j] if colors is None else colors[j]
                    log_dirs = self.filter_log_dirs(pattern='.*%s.*' % (p), **kwargs)
                    x, y = self.load_results(log_dirs, **kwargs)
                    variance = np.sum([np.std(y[:, t]) ** 2 for t in np.arange(y.shape[1])]) / 101
                    print(f'variance {variance}')
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
                        raise NotImplementedError
                plt.xticks([0, int(5e5)], ['0', r'$5 \times 10^5$'], fontsize=fontsize)
                # plt.title(titles[ind], fontsize=fontsize)
                plt.tick_params(axis='y', labelsize=fontsize)
                plt.xlabel('Steps', fontsize=fontsize)
                plt.ylabel('Return', rotation='horizontal', fontsize=fontsize, labelpad=20)
                plt.legend(fontsize=legend_size, loc="upper left")

    plot_games(plotter,
               titles=titles,
               agg='mean',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='episodic_return_test',
               root=log_root,
               interpolation=0,
               window=0,
               )

    plt.tight_layout()
    plt.savefig('%s/cartpole_%s.pdf' % (img_root, group), bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    # plot_baird(0, False)
    # plot_baird(1, False)
    # plot_baird(2, False)
    # plot_baird(3, False)
    # read_variance(0, False)
    # read_variance(1, False)
    # read_variance(2, False)

    # plot_cartpole(0, False)
    # plot_cartpole(1, False)