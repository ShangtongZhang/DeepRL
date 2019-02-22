import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_rl import *

def plot(**kwargs):
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('window', 100)
    kwargs.setdefault('down_sample', True)
    kwargs.setdefault('root', '../large_logs/two-circle')
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_tf_results(names, 'p0', kwargs['window'], align=True)
    print('')

    if kwargs['average']:
        color = kwargs['color']
        x = np.asarray(data[0][0])
        y = [y for _, y in data]
        y = np.asarray(y)
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


def extract_heatmap_data():
    coef = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    results = {}
    plotter = Plotter()
    root = '../large_logs/two-circle'
    for gamma_hat in coef:
        for lam2 in coef:
            print(gamma_hat, lam2)
            pattern = 'alg_GACE-gamma_hat_%s-lam2_%s-run' % (gamma_hat, lam2)
            pattern = translate(pattern)
            pattern = '.*%s.*' % (pattern)
            print(pattern)
            dirs = plotter.load_log_dirs(pattern, root=root)
            data = plotter.load_tf_results(dirs, 'p0', align=True)
            _, y = zip(*data)
            y = np.asarray(y)
            print(y.shape)
            y = y[:, :-100].mean()
            results[(gamma_hat, lam2)] = y

    print(results)
    with open('data/two-circle.bin', 'wb') as f:
        pickle.dump(results, f)


def two_circle_heatmap():
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    with open('data/two-circle.bin', 'rb') as f:
        data = pickle.load(f)
    print(data)
    coef = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    p0 = np.zeros((len(coef[:-1]), len(coef)))
    for i in range(len(coef[:-1])):
        for j in range(len(coef)):
            p0[i, j] = data[(coef[i], coef[j])]
    ax = sns.heatmap(p0, cmap='YlGnBu')
    ax.set_xticks(np.arange(0, 11) + 0.5)
    ax.set_xticklabels(['%s' % x for x in coef])
    ax.set_yticks(np.arange(0, 10) + 0.5)
    ax.set_yticklabels(['%s' % x for x in coef[:-1]], rotation='horizontal')
    plt.xlabel(r'$\lambda_2$')
    plt.ylabel(r'$\hat{\gamma}$', rotation='horizontal')
    plt.show()


def plot_mdp():
    kwargs = {
        'window': 0,
        'top_k': 0,
        'max_timesteps': int(1e5),
        'average': True,
        'x_interval': 100
    }

    patterns = [
        'alg_ACE-run',
        'alg_GACE-gamma_hat_0.9-lam2_0-run',
        'alg_GACE-gamma_hat_0.9-lam2_0.2-run',
        'alg_GACE-gamma_hat_0.9-lam2_0.4-run',
        'alg_GACE-gamma_hat_0.9-lam2_0.6-run',
        'alg_GACE-gamma_hat_0.9-lam2_0.8-run',
        'alg_GACE-gamma_hat_0.9-lam2_1-run',
    ]

    for i, p in enumerate(patterns):
        p = translate(p)
        plot(pattern='.*%s.*' % (p), color=i, **kwargs)
    plt.show()


if __name__ == '__main__':
    # two_circle_heatmap()
    plot_mdp()