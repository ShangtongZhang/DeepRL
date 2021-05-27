import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
import os

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
from deep_rl import *

img_root = '/Users/Shangtong/GoogleDrive/Paper/target-network/img'
fontsize = 22

def plot_baird(group):
    log_root = './log/target-net'
    plotter = Plotter()

    # 'BairdControl-v0.*lr_t_0.001-ridge_0-softmax_mu_False'
    ridges = [0, 0.01, 0.1]
    if group == 0:
        patterns = [
            ['BairdPrediction-v0.*lr_t_1-ridge_%s-' % (ridge) for ridge in ridges],
            ['BairdPrediction-v0.*lr_t_0.01-ridge_%s-' % (ridge) for ridge in ridges],
        ]
    elif group == 1:
        patterns = [
            ['BairdControl-v0.*lr_t_1-ridge_%s-softmax_mu_False' % (ridge) for ridge in ridges],
            ['BairdControl-v0.*lr_t_0.001-ridge_%s-softmax_mu_False' % (ridge) for ridge in ridges],
        ]
    elif group == 2:
        patterns = [
            ['BairdControl-v0.*lr_t_1-ridge_%s-softmax_mu_True' % (ridge) for ridge in ridges],
            ['BairdControl-v0.*lr_t_0.001-ridge_%s-softmax_mu_True' % (ridge) for ridge in ridges],
        ]
    else:
        raise NotImplementedError
    labels = [
        # [r'w/o target net, $\eta=%s$' % (ridge) for ridge in ridges],
        # [r'w/ target net, $\eta=%s$' % (ridge) for ridge in ridges],
        [r'standard, $\eta=%s$' % (ridge) for ridge in ridges],
        [r'ours, $\eta=%s$' % (ridge) for ridge in ridges],
    ]

    titles = ['']

    upper_limit = 10 ** 5

    def plot_games(self, titles, **kwargs):
        kwargs.setdefault('agg', 'mean')
        _, axes = plt.subplots(2, 1, sharex=True)

        for i, ax in enumerate(axes):
            for j, p in enumerate(patterns[i]):
                label = labels[i][j]
                color = self.COLORS[(1 - i) * 3 + j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*' % (p), **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                y = np.nan_to_num(y, nan=upper_limit)
                y[np.where(y >= upper_limit)] = upper_limit * 2
                assert y.shape[0] == 30
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se')
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std', ax=ax)
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    raise NotImplementedError
        ax1, ax2 = axes
        ax1.set_ylim(1e4, 1e5)
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False, labelsize=fontsize)  # don't put tick labels at the top
        ax1.set_yticks([1e4, 1e5])
        ax1.set_yticklabels([r'$10^4$', r'$10^5$'], fontsize=fontsize)
        ax1.legend(fontsize=fontsize)

        ax2.set_xticks([0, 100])
        ax2.set_xticklabels(['0', r'$5 \times 10^5$'], fontsize=fontsize)
        ax2.set_xlabel('Steps', fontsize=fontsize)
        ax2.legend(fontsize=fontsize)
        ax2.xaxis.tick_bottom()

        if group == 0:
            y_lim = 150
            y_label = r'$||Xw - v_\pi||$'
        elif group == 1:
            y_lim = 200
            y_label = r'$||Xw - q_*||$'
        elif group == 2:
            y_lim = 100
            y_label = r'$||Xw - q_*||$'
        else:
            raise NotImplementedError

        ax1.set_ylabel(y_label, fontsize=fontsize)
        ax2.set_ylim(0, y_lim)
        ax2.set_yticks([0, y_lim])
        ax2.set_yticklabels(['0', '%s' % y_lim], fontsize=fontsize)

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plot_games(plotter,
               titles=titles,
               agg='mean_std',
               downsample=0,
               right_align=True,
               right_most=0,
               tag='error',
               root=log_root,
               interpolation=0,
               window=0,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/baird_results_%d.pdf' % (img_root, group), bbox_inches='tight')


def plot_kolter():
    eps = 0.01
    p0 = (2961 + 45240 * eps + 40400 * eps ** 2) / (4141 + 84840 * eps + 40400 * eps ** 2)
    P = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    gamma = 0.99
    v = torch.tensor([1, 1.05]).view(-1, 1)
    I = torch.eye(2)
    r = (I - gamma * P) @ v
    X = torch.tensor([1, 1.05 + eps]).view(-1, 1)
    p1 = p0 - 0.00001
    p2 = p0 + 0.00001
    ps = np.concatenate([np.linspace(0.01, p1, 100),
                         np.linspace(p1, p2, 100),
                         np.linspace(p2, 0.99, 100),
                         ])
    etas = [0, 0.01, 0.02, 0.03]
    labels = [r'$w^*_{\eta = %s}$' % eta for eta in etas]
    errors = np.zeros((len(labels), len(ps)))
    for i, p in enumerate(ps):
        for j, eta in enumerate(etas):
            D = torch.diag(torch.tensor([p, 1 - p])).float()
            A = X.t() @ D @ (I - gamma * P) @ X + eta
            b = X.t() @ D @ r
            w = A.inverse() @ b
            errors[j, i] = (X @ w - v).norm(2)
    for i in range(len(labels)):
        plt.plot(ps, errors[i], label=labels[i])
    plt.xticks([0, 1], ['0', '1'], fontsize=fontsize)
    plt.xlabel(r'$d_\mu(s_1)$', fontsize=fontsize)
    plt.ylabel(r'$||Xw^*_\eta - v_\pi||$', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.yscale('log')
    plt.legend(fontsize=fontsize)
    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/kolter.pdf' % (img_root), bbox_inches='tight')
    exit(0)


if __name__ == '__main__':
    mkdir('images')
    # plot_baird(0)
    plot_baird(1)
    # plot_baird(2)
    # plot_kolter()
