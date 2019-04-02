import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
from deep_rl import *


def plot_ppo():
    plotter = Plotter()
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        # 'Swimmer-v2',
        # 'Reacher-v2',
        # 'dm-finger',

        'dm-walker',
        'dm-cartpole-b',
        # 'dm-cartpole-s',
        'dm-reacher',
        # 'dm-fish',
    ]


    patterns = [
        # 'remark_ppo',
        'PPO',
        'ASC',
    ]

    labels = [
        'PPO',
        'ASC',
    ]

    def top_k_measure(x):
        return np.mean(x)
        # return np.mean(x[400: 500])

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./log/ASC',
                       interpolation=100,
                       window=20,
                       top_k=0,
                       top_k_measure=top_k_measure,
                       )

    plt.show()
    # plt.tight_layout()
    # plt.savefig('images/PPO.png', bbox_inches='tight')


def plot_ddpg():
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
    ]

    labels = [
        'DDPG'
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TEST,
                       root='./data/benchmark/ddpg',
                       interpolation=0,
                       window=0,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/DDPG.png', bbox_inches='tight')


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

    plt.show()
    plt.tight_layout()
    # plt.savefig('images/Breakout.png', bbox_inches='tight')


def plot_misc():
    plotter = Plotter()
    games = [
        # 'BreakoutNoFrameskip-v4',
        'AsterixNoFrameskip-v4',
    ]

    patterns = [
        'beta_reg_0-remark_OC-run',
        'beta_reg_0\.01-remark_OC-run',
    ]

    labels = patterns

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=100,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./tf_log',
                       interpolation=0,
                       window=100,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/Breakout.png', bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    plot_ppo()
    # plot_ddpg()
    # plot_atari()
    # plot_misc()
