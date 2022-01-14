import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

def compute_v(N, gamma):
    v = np.zeros(N)
    # np.random.seed(0)
    np.random.seed()
    # rewards = np.random.rand(N) * 2 - 1
    rewards = np.random.rand(N)
    v[-1] = 0
    for i in reversed(range(N - 1)):
        v[i] = rewards[i] + gamma * v[i + 1]
    return v


def simulate_error(gamma, alpha, normalized=True):
    N = 100
    K = 30
    trials = 10000
    # trials = 1
    errors = []
    for t in range(trials):
        v = compute_v(N, gamma).reshape((N, 1))
        X = np.random.rand(N, K) * 4 - 2
        X = np.tanh(X)
        indices = np.arange(N)
        np.random.shuffle(indices)
        pos = int(N * alpha)
        indices[:pos] = np.random.choice(indices[pos: ], pos)
        X = X[indices]
        X = X + np.random.randn(N, K) * 0.1
        # X[:pos] = X[:pos] + np.random.randn(pos, K) * 0.1
        w = np.linalg.lstsq(X, v.flatten())[0]
        v_hat = X @ np.reshape(w, (K, 1))
        error = np.linalg.norm(v_hat - v)
        if normalized:
            error = error / np.linalg.norm(v)
        errors.append(error)
    print(gamma, np.mean(errors), np.std(errors) / np.sqrt(N))
    return np.mean(errors), np.std(errors)


def expts(normalized=True):
    gammas = np.linspace(0, 2, 20) * 0.1 + 0.8
    errors = []
    stds = []
    alphas = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    for alpha in alphas:
        infos = [simulate_error(gamma, alpha, normalized) for gamma in gammas]
        errors.append(np.array([info[0] for info in infos]))
        stds.append(np.array([info[1] for info in infos]))
    # ylabel = r'$\frac{||X(X^\top X)^{-1}X^\top v_\gamma - v_\gamma||}{||v_\gamma||}$'
    fontsize = 20
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5))
    for i, ax in enumerate(axes):
        m_x = errors[i]
        e_x = stds[i]
        ax.plot(gammas, errors[i])
        ax.fill_between(gammas, m_x + e_x, m_x - e_x, alpha=0.3)
        ax.set_xlabel(r'$\gamma$', fontsize=fontsize)
        ax.set_xticks([0.8, 0.9, 1])
        ylabel = 'NRE' if normalized else 'RE'
        if not i:
            ax.set_ylabel(ylabel, rotation='horizontal', labelpad=25, fontsize=fontsize)
        ax.set_title(f'{alphas[i] * 100}% aliased'.replace('%', r'\%'), fontsize=fontsize)
    plt.tight_layout()
    # plt.show()
    plt.savefig('/Users/Shangtong/GoogleDrive/Paper/discounting/img/state_aliasing_%s.pdf' % (normalized), bbox_inches='tight')


if __name__ == '__main__':
    # expts(True)
    expts(False)
