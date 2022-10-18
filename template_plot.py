import matplotlib
from matplotlib import scale
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
from deep_rl import *

plt.rc('text', usetex=True)
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
img_root = '/Users/Shangtong/GoogleDrive/Paper/stochastic-policy-pseudo-gradients/img'
fontsize = 22

def plot_chain():
    plotter = Plotter()
    games = [
        'Chain-v0',
    ]

    num_states = 9 
    eps_reg = np.array([0.6, 1, 2, 4, 8, 16, 32]) / 16 

    patterns = [
        f'eps_reg_{eps}-num_states_{num_states}-.*sac_True' for eps in eps_reg
    ]

    labels = patterns

    def plot_games(self, games, **kwargs):
        variance = {}
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        plt.figure(figsize=(l * 5, 5))
        for i, game in enumerate(games):
            variance[i] = []
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(
                    pattern='.*%s.*%s' % (game, p), **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                var = np.mean(np.std(y, axis=0))
                variance[i].append(var)
                if kwargs['downsample']:
                    indices = np.linspace(
                        0, len(x) - 1, kwargs['downsample']).astype(np.int)
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
            plt.xlabel('steps')
            if not i:
                plt.ylabel(kwargs['tag'])
            plt.title(game)
            plt.legend()
        print(variance)

    plot_games(
        plotter,
        games=games,
        patterns=patterns,
        agg='mean',
        downsample=0,
        labels=labels,
        right_align=True,
        tag=plotter.RETURN_TEST,
        root='./log/off-pac-kl/tmp4',
        interpolation=0,
        window=100,
    )

    plt.show()


def plot_paper(sac=False):
    plotter = Plotter()
    game = 'Chain-v0'

    num_states = [6, 7, 8, 9]
    eps_reg = np.array([0.5, 1, 2, 8, 32]) / 16 
    eps_exp = [int(np.round(math.log(eps, 2))) for eps in eps_reg]

    patterns = [[
        f'eps_reg_{eps}-num_states_{num_s}-.*sac_{sac}' for eps in eps_reg
    ] for num_s in num_states]

    labels = [r'$\epsilon_\lambda=2^{%s}$' % (eps) for eps in eps_exp]

    if sac:
        scales = [1, 1, 1, 1, 1]
    else:
        scales = [1, 1, 1, 1, 1]


    def plot_games(self, **kwargs):
        variance = {}
        kwargs.setdefault('agg', 'mean')
        # import matplotlib.pyplot as plt
        l = len(num_states)
        plt.figure(figsize=(l * 5, 5))
        for i, num_s in enumerate(num_states):
            variance[i] = []
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(patterns[i]):
                label = labels[j] 
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(
                    pattern='.*%s.*%s' % (game, p), **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                var = np.mean(np.std(y, axis=0))
                variance[i].append(var)
                if kwargs['downsample']:
                    indices = np.linspace(
                        0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se', scale=scales[i])
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std', scale=scales[i])
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color, scale=scales[i])
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            optimal_r = 0.99 ** (num_s - 1)
            plt.plot(x, np.ones_like(x) * optimal_r, color='black', linestyle=(0, (5, 10)))
            plt.xticks([0, x[-1]], ['0', r'$2 \times 10^6$'], fontsize=fontsize)
            plt.title(rf'$N={num_s}$', fontsize=fontsize)
            plt.xlabel('Steps', fontsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            if not i:
                plt.ylabel(r'$J(\pi;p_0)$', fontsize=fontsize, rotation='horizontal', labelpad=35)
                plt.legend(fontsize=fontsize - 3)

        print(variance)

    plot_games(
        plotter,
        agg='mean',
        downsample=0,
        right_align=True,
        tag=plotter.RETURN_TEST,
        root='./log/off-pac-kl',
        interpolation=0,
        window=10,
    )

    plt.tight_layout()
    plt.savefig(f'{img_root}/sac_{sac}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    # plot_ppo()
    # plot_ddpg_td3()
    # plot_atari()
    # plot_chain()

    plot_paper(sac=False)
    # plot_paper(sac=True)

