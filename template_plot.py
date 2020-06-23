import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from deep_rl import *


def plot_games(self, games, **kwargs):
    kwargs.setdefault('agg', 'mean')
    import matplotlib.pyplot as plt
    l = len(games)
    plt.figure(figsize=(l * 5, 5))
    for i, game in enumerate(games):
        plt.subplot(1, l, i + 1)
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
        plt.xlabel('steps')
        if not i:
            plt.ylabel(kwargs['tag'])
        plt.title(game)
        plt.ylim([0, 300])
        # plt.ylim([0, 10])
        plt.legend()


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


PATH = '/Users/Shangtong/GoogleDrive/Paper/cof-pac/img'


def emphasis_figures():
    plotter = Plotter()
    games = [
        'OriginalBaird-v0',
        'OneHotBaird-v0',
        'ZeroHotBaird-v0',
        'AliasedBaird-v0',
    ]

    def get_info(pi_solid):
        str_pi_solid = ('%s' % pi_solid).replace('.', '\.')
        p = {}
        if pi_solid == 0.1:
            p['OriginalBaird-v0'] = 'lr_m_0\.025-pi_solid_%s-run' % (str_pi_solid)
            p['OneHotBaird-v0'] = 'lr_m_0\.2-pi_solid_%s-run' % (str_pi_solid)
            p['ZeroHotBaird-v0'] = 'lr_m_0\.0125-pi_solid_%s-run' % (str_pi_solid)
            p['AliasedBaird-v0'] = 'lr_m_0\.05-pi_solid_%s-run' % (str_pi_solid)

            labels = [
                r'GEM-Original($\alpha=0.1 \times 2^{-2}$)',
                r'GEM-OneHot($\alpha=0.1 \times 2^1$)',
                r'GEM-ZeroHot($\alpha=0.1 \times 2^{-3}$)',
                r'GEM-Aliased($\alpha=0.1 \times 2^{-1}$)',
            ]

            xticks = ([0, 1000], ['0', r'$10^6$'])
            title = r'$\pi(solid | \cdot)=0.1$'

        elif pi_solid == 0.3:
            p['OriginalBaird-v0'] = 'lr_m_0\.00625-max_steps_2000000-pi_solid_%s-run' % (str_pi_solid)
            p['OneHotBaird-v0'] = 'lr_m_0\.05-max_steps_2000000-pi_solid_%s-run' % (str_pi_solid)
            p['ZeroHotBaird-v0'] = 'lr_m_0\.0125-max_steps_2000000-pi_solid_%s-run' % (str_pi_solid)
            p['AliasedBaird-v0'] = 'lr_m_0\.0125-max_steps_2000000-pi_solid_%s-run' % (str_pi_solid)

            labels = [
                r'GEM-Original($\alpha=0.1 \times 2^{-4}$)',
                r'GEM-OneHot($\alpha=0.1 \times 2^{-1}$)',
                r'GEM-ZeroHot($\alpha=0.1 \times 2^{-3}$)',
                r'GEM-Aliased($\alpha=0.1 \times 2^{-3}$)',
            ]

            xticks = ([0, 2000], ['0', r'$2 \times 10^6$'])
            title = r'$\pi(solid | \cdot)=0.3$'
        else:
            raise NotImplementedError
        return dict(
            patterns=p,
            labels=labels,
            xticks=xticks,
            title=title,
        )

    fontsize = 30
    import matplotlib.pyplot as plt
    pi = [0.1, 0.3]
    l = len(pi)
    plt.figure(figsize=(l * 5, 5))
    for j, pi_solid in enumerate(pi):
        plt.subplot(1, l, j + 1)
        info = get_info(pi_solid)
        for i, game in enumerate(games):
            log_dirs = plotter.filter_log_dirs(
                pattern='.*%s.*%s' % (game, info['patterns'][game]),
                root='./log/gem-baird',
            )
            x, y = plotter.load_results(
                log_dirs,
                interpolation=0,
                tag='gem_loss',
            )
            indices = np.linspace(0, len(x) - 1, 100).astype(np.int)
            x = x[indices]
            y = y[:, indices]
            plotter.plot_mean(y, x, label=info['labels'][i], color=plotter.COLORS[i], error='std', marker=plotter.MARKERS[i], markevery=10)

        x, y = plotter.load_results(
            log_dirs,
            interpolation=0,
            tag='trace_loss',
        )
        indices = np.linspace(0, len(x) - 1, 100).astype(np.int)
        x = x[indices]
        y = y[:, indices]
        plotter.plot_mean(y, x, label='followon trace', color=plotter.COLORS[i + 1], error='std', marker=plotter.MARKERS[i + 1], markevery=10)

        plt.xlabel('steps', fontsize=fontsize)
        ylim = [0, 140]
        plt.ylim(ylim)
        if not j:
            plt.ylabel('emphasis error', fontsize=fontsize)
            plt.yticks(ylim, ylim, fontsize=fontsize)
        else:
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
        plt.xticks(*info['xticks'], fontsize=fontsize)
        plt.legend(fontsize=15)
        plt.title(info['title'], fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('%s/emphasis_error.pdf' % (PATH), bbox_inches='tight')


def gem_etd_figures():
    plotter = Plotter()
    games = [
        'OriginalBaird-v0',
        'OneHotBaird-v0',
        'ZeroHotBaird-v0',
        'AliasedBaird-v0',
    ]

    titles = [
        'Original',
        'OneHot',
        'ZeroHot',
        'Aliased',
    ]

    lrs = 0.1 * np.power(2.0, -np.arange(0, 20))
    lr_strs = {}
    for lr, p in zip(lrs, np.arange(0, 20)):
        lr_strs[lr] = p

    def escape(x):
        return ('%s' % x).replace('.', '\.')

    m_types = ['gem', 'trace']

    def get_data():
        data = {}
        ranks = 4
        for game in games:
            data[game] = {}
            for m in m_types:
                data[game][m] = {}
                auc = []
                ys = []
                for lr in lrs:
                    p = 'etd_True-lr_etd_%s-lr_m_0\.025-m_type_%s-max_steps_100000-pi_solid_0\.05-run' % (escape(lr), m)
                    log_dirs = plotter.filter_log_dirs(pattern='.*%s.*%s' % (game, p),
                                                       root='./log/gem-etd-baird')
                    x, y = plotter.load_results(log_dirs,
                                                tag='RMSVE',
                                                right_align=False,
                                                interpolation=0,
                                                window=0)
                    ys.append(y)
                    auc.append(y.mean(1).sum())
                for rank in range(ranks):
                    best_lr_ind = np.argsort(auc)[rank]
                    data[game][m][rank] = [lrs[best_lr_ind], x, ys[best_lr_ind]]
        return data

    data = get_data()

    fontsize = 30
    import matplotlib.pyplot as plt
    l = len(games)
    plt.figure(figsize=(l * 4.5, 5))
    # plt.figure(figsize=(2 * 5, 2 * 5))
    ranks = [3, 1, 2, 1]
    for i, game in enumerate(games):
        plt.subplot(1, l, i + 1)
        # plt.subplot(2, 2, i + 1)
        for j, m in enumerate(m_types):
            lr, x, y = data[game][m][0]
            if m == 'gem':
                label = r'Gem-ETD(0)($\alpha_2=0.1\times 2^{-%d}$)' % (lr_strs[lr])
            else:
                label = r'ETD(0)($\alpha=0.1\times 2^{-%d}$)' % (lr_strs[lr])
            color = plotter.COLORS[j]
            marker = plotter.MARKERS[j]
            plotter.plot_mean(y, x, label=label, color=color, error='std', marker=marker, markevery=10)

            if m == 'gem':
                lr, x, y = data[game][m][ranks[i]]
                color = plotter.COLORS[len(m_types)]
                label = r'Gem-ETD(0)($\alpha_2=0.1\times 2^{-%d}$)' % (lr_strs[lr])
                plotter.plot_mean(y, x, label=label, color=color, error='std', linestyle=':', marker=None, markevery=1)

        plt.xlabel('steps', fontsize=fontsize)
        plt.xticks([0, 100], ['0', r'$10^5$'], fontsize=fontsize)
        if not i:
            # plt.ylabel('RMSVE', rotation='horizontal', fontsize=30)
            plt.ylabel('RMSVE', fontsize=fontsize)
            plt.ylim([0, 300])
            plt.yticks([0, 300], [0, 300], fontsize=fontsize)
        else:
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
        plt.title(titles[i], fontsize=fontsize)
        plt.legend(fontsize=15)

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/RMSVE.pdf' % (PATH), bbox_inches='tight')


def plot_mujoco():
    plotter = Plotter()
    games = ['Reacher-v2']

    patterns = [
        'algo_cof-pac',
        'algo_ace',
        'TD3-random',
    ]

    labels = [
        'COF-PAC',
        'ACE',
        'TD3',
    ]

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        plt.figure(figsize=(l * 5, 4))
        for i, game in enumerate(games):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                marker = self.MARKERS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, p), **kwargs)
                if j == 2:
                    kwargs['tag'] = 'episodic_return'
                x, y = self.load_results(log_dirs, **kwargs)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                x = x[:50]
                y = y[:, :50]
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
            fontsize = 30
            plt.xlabel('steps', fontsize=fontsize)
            plt.xticks([0, int(5e4)], ['0', r'$5\times 10^4$'], fontsize=fontsize)
            if not i:
                plt.ylabel(r'$J(\pi)$', rotation='horizontal', fontsize=fontsize)
                ylim = [-20, -3]
                plt.ylim(ylim)
                plt.yticks(ylim, ylim, fontsize=fontsize)
            plt.title(game, fontsize=fontsize)
            plt.legend(fontsize=15)

    kwargs = dict(
        patterns=patterns,
        agg='mean_std',
        downsample=0,
        labels=labels,
        right_align=False,
        tag=plotter.RETURN_TEST,
        root='./log/cof-pac-mujoco',
        interpolation=0,
        window=0,
    )

    plot_games(plotter, games, **kwargs)

    # plt.show()
    plt.tight_layout()
    plt.savefig('%s/mujoco.pdf' % (PATH), bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    # plot_ppo()
    # plot_ddpg_td3()
    # plot_atari()

    # emphasis_figures()
    gem_etd_figures()
    # plot_mujoco()
