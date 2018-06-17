import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from deep_rl import *

def compute_stats(**kwargs):
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=1e8)
    max_rewards = []
    for x, y in data:
        max_rewards.append(np.max(y))
    return np.mean(max_rewards), np.std(max_rewards) / np.sqrt(len(max_rewards))

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=50, max_timesteps=kwargs['max_timesteps'])
    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x, y = plotter.average(data, 100, kwargs['max_timesteps'], top_k=kwargs['top_k'])
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i==0 else '')
    plt.legend()
    # plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

def ddpg_plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('max_timesteps', 1e8)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=0, max_timesteps=kwargs['max_timesteps'])
    data = [y[: len(y) // kwargs['rep'] * kwargs['rep']] for x, y in data]
    min_y = np.min([len(y) for y in data])
    data = [y[ :min_y] for y in data]
    new_data = []
    for y in data:
        y = np.reshape(np.asarray(y), (-1, kwargs['rep'])).mean(-1)
        x = np.arange(y.shape[0]) * kwargs['x_interval']
        new_data.append([x, y])
    data = new_data

    print('')

    figure = kwargs['figure']
    color = kwargs['color']
    plt.figure(figure)
    if kwargs['average']:
        x = data[0][0]
        y = [entry[1] for entry in data]
        # y = np.transpose(np.stack(y))
        y = np.stack(y)
        name = names[0].split('/')[-1]
        sns.tsplot(y, x, condition=name, color=Plotter.COLORS[color])
        plt.title(names[0])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            plt.plot(x, y, color=Plotter.COLORS[i], label=name if i==0 else '')
    plt.legend()
    # plt.ylim([-200, 1400])
    # plt.ylim([-200, 2500])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()

if __name__ == '__main__':
    kwargs = {
        'x_interval': int(1e4),
        'rep': 20,
        'average': True
    }
    # patterns = [
    #     'per_episode_decay',
    #     'per_episode_random',
    #     'per_step_decay',
    #     'per_step_random'
    # ]
    games = [
        'RoboschoolAnt-v1',
        # 'RoboschoolWalker2d-v1',
        # 'RoboschoolHopper-v1',
        # 'RoboschoolHalfCheetah-v1',
        # 'RoboschoolReacher-v1',
        # 'RoboschoolHumanoid-v1',
    ]
    patterns = [
        'original',
        'q_ddpg',
        'ucb_ddpg_c0',
        'ucb_ddpg_c10',
        'ucb_ddpg_c50',
    ]
    for j, game in enumerate(games):
        for i, p in enumerate(patterns):
            ddpg_plot(pattern='.*log/ucb-ddpg/ucb-%s.*%s.*' % (game, p), figure=j, color=i, **kwargs)
    plt.show()
