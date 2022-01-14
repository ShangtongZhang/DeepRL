import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from deep_rl import *
from multiprocessing import Pool
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'


def plot_ppo():
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Reacher-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
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
                       root='./data/benchmark/mujoco',
                       interpolation=100,
                       window=10,
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
        'Ant-v2',
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
                       root='./data/benchmark/mujoco',
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
        'remark_quantile',
    ]

    labels = [
        'A2C',
        'C51',
        'DQN',
        'N-Step DQN',
        'OC',
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


def plot_ppo_patterns(patterns, labels, filename, dir, field='episodic_return_train', params={}, patterns_fn=None):
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]
    if labels is None:
        labels = patterns

    if isinstance(field, list):
        fields = field
    else:
        fields = [field] * len(patterns)

    patterns = [[p, ' '] if isinstance(p, str) else p for p in patterns]

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        # plt.figure(figsize=(l * 5, 5))
        plt.figure(figsize=(3 * 5, 2 * 5))
        for i, game in enumerate(games):
            # plt.subplot(1, l, i + 1)
            plt.subplot(2, 3, i + 1)
            if patterns_fn is None:
                patterns = kwargs['patterns']
                labels = kwargs['labels']
            else:
                patterns, labels = patterns_fn(game)
            for j, p in enumerate(patterns):
                label = labels[j]
                color = self.COLORS[j]
                positve_p, negative_p = (p, ' ') if isinstance(p, str) else p
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, positve_p),
                                                negative_pattern='.*%s.*' % (negative_p),
                                                **kwargs)
                kwargs['tag'] = fields[j]
                x, y = self.load_results(log_dirs, **kwargs)
                assert y.shape[0] == 10 or y.shape[0] == 3
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                assert y.shape[0] == 10 or y.shape[0] == 3
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
            if i in [0, 3]:
                plt.ylabel(kwargs['ylabel'])
            if i >= 3:
                plt.xlabel('steps')
            plt.yscale(kwargs['yscale'])
            plt.title(game)
            plt.legend()

    params.setdefault('downsample', 100)
    params.setdefault('window', 10)
    params.setdefault('interpolation', 100)
    params.setdefault('ylabel', field)
    params.setdefault('yscale', 'linear')
    plot_games(plotter,
               games=games,
               patterns=patterns,
               agg='mean',
               labels=labels,
               right_align=False,
               tag=field,
               root=dir,
               **params,
               )

    # plt.show()
    plt.tight_layout()
    plt.savefig(
        './images/%s.pdf' % (filename), bbox_inches='tight')


def plot_impl():
    data = {}
    data[0] = [
        [
            'discount_0.99.*gae_tau_0-lr_ratio_1.*use_gae_True',
            'discount_0.99.*lr_ratio_1.*use_gae_False',
            'discount_0.99.*gae_tau_0-lr_ratio_0.5.*use_gae_True',
            'discount_0.99.*lr_ratio_0.5.*use_gae_False',
            'discount_0.99.*gae_tau_0-lr_ratio_2.*use_gae_True',
            'discount_0.99.*lr_ratio_2.*use_gae_False',
        ],
        [
            r'$\gamma=0.99$, lr_multiplier=1, TD advantage',
            r'$\gamma=0.99$, lr_multiplier=1, MC advantage',
            r'$\gamma=0.99$, lr_multiplier=0.5, TD advantage',
            r'$\gamma=0.99$, lr_multiplier=0.5, MC advantage',
            r'$\gamma=0.99$, lr_multiplier=2, TD advantage',
            r'$\gamma=0.99$, lr_multiplier=2, MC advantage',
        ],
        r'ppo-gae-0.99',
        './pt/ppo1',
    ]

    data[1] = [
        [
            'discount_0.97.*gae_tau_0-lr_ratio_1.*use_gae_True',
            'discount_0.97.*lr_ratio_1.*use_gae_False',
            'discount_0.97.*gae_tau_0-lr_ratio_0.5.*use_gae_True',
            'discount_0.97.*lr_ratio_0.5.*use_gae_False',
            'discount_0.97.*gae_tau_0-lr_ratio_2.*use_gae_True',
            'discount_0.97.*lr_ratio_2.*use_gae_False',
        ],
        [
            r'$\gamma=0.97$, lr_multiplier=1, TD advantage',
            r'$\gamma=0.97$, lr_multiplier=1, MC advantage',
            r'$\gamma=0.97$, lr_multiplier=0.5, TD advantage',
            r'$\gamma=0.97$, lr_multiplier=0.5, MC advantage',
            r'$\gamma=0.97$, lr_multiplier=2, TD advantage',
            r'$\gamma=0.97$, lr_multiplier=2, MC advantage',
        ],
        r'ppo-gae-0.97',
        './pt/ppo1',
    ]

    data[2] = [
        [
            'discount_1.*gae_tau_0-lr_ratio_1.*use_gae_True',
            'discount_1.*lr_ratio_1.*use_gae_False',
            'discount_1.*gae_tau_0-lr_ratio_0.5.*use_gae_True',
            'discount_1.*lr_ratio_0.5.*use_gae_False',
            'discount_1.*gae_tau_0-lr_ratio_2.*use_gae_True',
            'discount_1.*lr_ratio_2.*use_gae_False',
        ],
        [
            r'$\gamma=1$, lr_multiplier=1, TD advantage',
            r'$\gamma=1$, lr_multiplier=1, MC advantage',
            r'$\gamma=1$, lr_multiplier=0.5, TD advantage',
            r'$\gamma=1$, lr_multiplier=0.5, MC advantage',
            r'$\gamma=1$, lr_multiplier=2, TD advantage',
            r'$\gamma=1$, lr_multiplier=2, MC advantage',
        ],
        r'ppo-gae-1',
        './pt/ppo1',
    ]

    data[3] = [
        [
            'discount_0.99.*gae_tau_0-lr_ratio_0.5.*use_gae_True',
            'discount_0.97.*gae_tau_0-lr_ratio_0.5.*use_gae_True',
            'discount_1.*gae_tau_0-lr_ratio_0.5.*use_gae_True',
        ],
        [
            r'$\gamma=0.99$, lr_multiplier=0.5, TD advantage',
            r'$\gamma=0.97$, lr_multiplier=0.5, TD advantage',
            r'$\gamma=1$, lr_multiplier=0.5, TD advantage',
        ],
        r'ppo-gae-td_adv',
        './pt/ppo1',
    ]

    data[4] = [
        [
            'discount_0.99.*lr_ratio_0.5.*use_gae_False',
            'discount_0.97.*lr_ratio_0.5.*use_gae_False',
            'discount_1.*lr_ratio_0.5.*use_gae_False',
        ],
        [
            r'$\gamma=0.99$, lr_multiplier=0.5, MC advantage',
            r'$\gamma=0.97$, lr_multiplier=0.5, MC advantage',
            r'$\gamma=1$, lr_multiplier=0.5, MC advantage',
        ],
        'ppo-gae-mc_adv',
        './pt/ppo1',
    ]

    data[5] = [
        [
            'aux_False.*d_scheme_0.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_1.*discount_0.99.*remark_dppo',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_True',
        ],
        [
            'PPO',
            'DisPPO',
            'AuxPPO',
            'AuxPPO-sync',
        ],
        'ppo-discounting-ret',
        './pt/ppo2',
        'episodic_return_train',
    ]

    data[6] = [
        [
            'aux_False.*d_scheme_0.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_1.*discount_0.99.*remark_dppo',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_True',
        ],
        [
            'PPO',
            'DisPPO',
            'AuxPPO',
            'AuxPPO-sync',
        ],
        'ppo-discounting-dis-ret',
        './pt/ppo2',
        'discounted_return_train',
    ]

    data[7] = [
        [
            'aux_False.*d_scheme_0.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_1.*discount_0.99.*remark_dppo',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_True',
        ],
        [
            'PPO',
            'DisPPO',
            'DisPPO aux1',
            'DisPPO aux2',
        ],
        'ppo-discounting-avg',
        './pt/ppo2',
        'avg_reward',
    ]

    data[8] = [
        [
            'aux_False.*d_scheme_0.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_1.*discount_0.99.*remark_dppo',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_True',
        ],
        [
            'PPO',
            'DisPPO',
            'DisPPO aux1',
            'DisPPO aux2',
        ],
        'ppo-discounting-len',
        './pt/ppo2',
        'episode_len',
    ]

    data[9] = [
        [
            'H_16.*lr_ratio_0.5.*remark_fhppo.*use_gae_True',
            'H_16.*lr_ratio_1.*remark_fhppo.*use_gae_True',
            'H_16.*lr_ratio_2.*remark_fhppo.*use_gae_True',
            'H_16.*lr_ratio_0.5.*remark_fhppo.*use_gae_False',
            'H_16.*lr_ratio_1.*remark_fhppo.*use_gae_False',
            'H_16.*lr_ratio_2.*remark_fhppo.*use_gae_False',
        ],
        [
            'H_16.*lr_ratio_0.5.*remark_fhppo.*use_gae_True',
            'H_16.*lr_ratio_1.*remark_fhppo.*use_gae_True',
            'H_16.*lr_ratio_2.*remark_fhppo.*use_gae_True',
            'H_16.*lr_ratio_0.5.*remark_fhppo.*use_gae_False',
            'H_16.*lr_ratio_1.*remark_fhppo.*use_gae_False',
            'H_16.*lr_ratio_2.*remark_fhppo.*use_gae_False',
        ],
        'FHPPO_H_16',
        './pt/ppo2',
        'episodic_return_train',
    ]

    for i, H in enumerate([32, 64, 128, 256]):
        info = []
        tag = 'H_%d' % (H)
        info.append([p.replace('H_16', tag) for p in data[9][0]])
        info.append([p.replace('H_16', tag) for p in data[9][1]])
        info.append(data[9][2].replace('H_16', tag))
        info.append(data[9][3])
        info.append(data[9][4])
        data[10 + i] = info

    data[14] = [
        [
            'discount_0.99.*remark_ppo',
            'discount_0.97.*remark_ppo',
            'discount_1.*remark_ppo',
        ],
        [
            r'$\gamma=0.99$',
            r'$\gamma=0.97$',
            r'$\gamma=1$',
        ],
        'ppo_oracle_bootstrap',
        './pt/ppo2',
    ]

    data[15] = [
        [
            'discount_1',
            'discount_0.99',
        ],
        [
            r'$\gamma=1$',
            r'$\gamma=0.99$',
        ],
        'ppo_oracle_v',
        './pt/ppo_oracle',
    ]

    data[16] = [
        [
            'H_256-discount_1-gae_tau_0-lr_ratio_0.5-max_steps_1000000-remark_fhppo-use_gae_True',
            'H_128-discount_1-gae_tau_0-lr_ratio_0.5-max_steps_1000000-remark_fhppo-use_gae_True',
            'H_64-discount_1-gae_tau_0-lr_ratio_0.5-max_steps_1000000-remark_fhppo-use_gae_True',
            # 'bootstrap_with_oracle_False-discount_0.97-gae_tau_0-lr_ratio_0.5-max_steps_1000000-normalized_adv_True-remark_ppo-use_gae_True',
            'bootstrap_with_oracle_False-discount_0.99-gae_tau_0-lr_ratio_0.5-max_steps_1000000-normalized_adv_True-remark_ppo-use_gae_True',
            'bootstrap_with_oracle_False-discount_1-gae_tau_0-lr_ratio_0.5-max_steps_1000000-normalized_adv_True-remark_ppo-use_gae_True',
        ],
        [
            r'FHPPO (H=256, $\gamma=1$)',
            r'FHPPO (H=128, $\gamma=1$)',
            r'FHPPO (H=64, $\gamma=1$)',
            # r'PPO ($\gamma=0.97)',
            r'PPO ($\gamma=0.99)',
            r'PPO ($\gamma=1)',
        ],
        'ppo-fhppo-td-adv',
        './pt',

    ]

    data[17] = [
        [
            'aux_False.*d_scheme_0.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_1.*discount_0.99.*remark_dppo',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_True',
        ],
        [
            'PPO',
            'DisPPO',
            'AuxPPO',
            'AuxPPO-sync',
        ],
        'aux_ppo-ret_train',
        './log/discounting/ppo3',
        'episodic_return_train',
    ]

    data[18] = [
        [
            'aux_False.*d_scheme_0.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_1.*discount_0.99.*remark_dppo',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_1.*discount_0.99.*remark_dppo.*sync_aux_True',
        ],
        [
            'PPO',
            'DisPPO',
            'AuxPPO',
            'AuxPPO-sync',
        ],
        'aux_ppo-dis_ret_train',
        './log/discounting/ppo3',
        'discounted_return_train',
    ]

    data[19] = [
        [
            'aux_False.*d_scheme_no.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_unbias.*discount_0.99.*remark_dppo',
            'aux_True.*d_scheme_unbias.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_unbias.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_unbias.*discount_0.99.*remark_dppo.*sync_aux_True',
            'aux_True.*d_scheme_unbias.*discount_0.99.*remark_dppo.*sync_aux_True',
        ],
        [
            'PPO',
            'DisPPO',
            r'AuxPPO $\pi$',
            r'AuxPPO $\pi^\prime$',
            r'AuxPPO-sync $\pi$',
            r'AuxPPO-sync $\pi^\prime$',
        ],
        'aux_ppo-ret_test',
        './log/discounting/aux_ppo',
        [
            'main_episodic_return_test',
            'main_episodic_return_test',
            'main_episodic_return_test',
            'aux_episodic_return_test',
            'main_episodic_return_test',
            'aux_episodic_return_test',
        ],
        dict(downsample=0, window=0, interpolation=0, ylabel='undiscounted_test_score')
    ]

    data[20] = [
        [
            'aux_False.*d_scheme_no.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_unbias.*discount_0.99.*remark_dppo',
            'aux_True.*d_scheme_unbias.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_unbias.*discount_0.99.*remark_dppo.*sync_aux_False',
            'aux_True.*d_scheme_unbias.*discount_0.99.*remark_dppo.*sync_aux_True',
            'aux_True.*d_scheme_unbias.*discount_0.99.*remark_dppo.*sync_aux_True',
        ],
        [
            'PPO',
            'DisPPO',
            r'AuxPPO $\pi$',
            r'AuxPPO $\pi^\prime$',
            r'AuxPPO-sync $\pi$',
            r'AuxPPO-sync $\pi^\prime$',
        ],
        'aux_ppo-dist_ret_test',
        './log/discounting/aux_ppo',
        [
            'main_discounted_return_test',
            'main_discounted_return_test',
            'main_discounted_return_test',
            'aux_discounted_return_test',
            'main_discounted_return_test',
            'aux_discounted_return_test',
        ],
        dict(downsample=0, window=0, interpolation=0, ylabel='discounted_test_score')
    ]

    data[21] = [
        [
            'aux_False.*d_scheme_no.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_unbias.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_comp_unbias.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_inv_linear.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_log_linear.*discount_0.99.*remark_dppo',
        ],
        [
            'PPO',
            r'PPO $\gamma^t$',
            r'PPO $1 - \gamma^{t+1}$',
            'PPO 1/(1+t)',
            'PPO log(1+t)',
        ],
        'discounting-ret_test',
        './log/discounting/aux_ppo',
        'main_episodic_return_test',
        dict(downsample=0, window=0, interpolation=0)
    ]

    data[22] = [
        [
            'aux_False.*d_scheme_no.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_unbias.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_comp_unbias.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_inv_linear.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_log_linear.*discount_0.99.*remark_dppo',
        ],
        [
            'PPO',
            r'PPO $\gamma^t$',
            r'PPO $1 - \gamma^{t+1}$',
            'PPO 1/(1+t)',
            'PPO log(1+t)',
        ],
        'discounting-dis_ret_test',
        './log/discounting/aux_ppo',
        'main_discounted_return_test',
        dict(downsample=0, window=0, interpolation=0)
    ]

    data[23] = [
        [
            'aux_False.*d_scheme_no.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_no.*discount_0.99.*remark_dppo',
            'aux_False.*d_scheme_no.*discount_0.99.*remark_dppo',
        ],
        [
            'undiscounted return',
            'discounted return',
            'average reward',
        ],
        'ppo_metric',
        './tf_log',
        [
            'episodic_return_train',
            'discounted_return_train',
            'avg_reward',
        ],
        dict(ylabel='score', yscale='log')
    ]

    for i, discount in enumerate([0.95, 0.97, 0.99, 0.995]):
        data[23 + 2 * i] = [
            [
                'aux_False.*d_scheme_no.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_unbias.*discount_%s.*remark_dppo' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_False' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_False' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_True' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                r'AuxPPO $\pi$',
                r'AuxPPO $\pi^\prime$',
                r'AuxPPO-sync $\pi$',
                r'AuxPPO-sync $\pi^\prime$',
            ],
            'aux_ppo-ret_test_%s' % (discount),
            './log/discounting/aux_ppo',
            [
                'main_episodic_return_test',
                'main_episodic_return_test',
                'main_episodic_return_test',
                'aux_episodic_return_test',
                'main_episodic_return_test',
                'aux_episodic_return_test',
            ],
            dict(downsample=0, window=0, interpolation=0, ylabel='undiscounted_test_score')
        ]

        data[23 + 2 * i + 1] = [
            [
                'aux_False.*d_scheme_no.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_unbias.*discount_%s.*remark_dppo' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_False' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_False' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_True' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                r'AuxPPO $\pi$',
                r'AuxPPO $\pi^\prime$',
                r'AuxPPO-sync $\pi$',
                r'AuxPPO-sync $\pi^\prime$',
            ],
            'aux_ppo-dis_ret_test_%s' % (discount),
            './log/discounting/aux_ppo',
            [
                'main_discounted_return_test',
                'main_discounted_return_test',
                'main_discounted_return_test',
                'aux_discounted_return_test',
                'main_discounted_return_test',
                'aux_discounted_return_test',
            ],
            dict(downsample=0, window=0, interpolation=0, ylabel='discounted_test_score')
        ]

    for i, discount in enumerate([0.95, 0.97, 0.99, 0.995]):
        data[31 + 3 * i] = [
            [
                'aux_False.*d_scheme_no.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_unbias.*discount_%s.*remark_dppo' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_False' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                r'AuxPPO $\pi$',
                r'AuxPPO-sync $\pi$',
            ],
            'aux_ppo-ret_train_%s' % (discount),
            './log/discounting/aux_ppo',
            'episodic_return_train',
        ]

        data[31 + 3 * i + 1] = [
            [
                'aux_False.*d_scheme_no.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_unbias.*discount_%s.*remark_dppo' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_False' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                r'AuxPPO $\pi$',
                r'AuxPPO-sync $\pi$',
            ],
            'aux_ppo-dis_ret_train_%s' % (discount),
            './log/discounting/aux_ppo',
            'discounted_return_train',
        ]

        data[31 + 3 * i + 2] = [
            [
                'aux_False.*d_scheme_no.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_unbias.*discount_%s.*remark_dppo' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_False' % (discount),
                'aux_True.*d_scheme_unbias.*discount_%s.*remark_dppo.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                r'AuxPPO $\pi$',
                r'AuxPPO-sync $\pi$',
            ],
            'aux_ppo-avg_reward_train_%s' % (discount),
            './log/discounting/aux_ppo',
            'avg_reward',
        ]

    for i, discount in enumerate([0.95, 0.97, 0.99, 0.995]):
        data[43 + i] = [
            [
                'aux_False.*d_scheme_no.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_no.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_no.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_unbias.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_unbias.*discount_%s.*remark_dppo' % (discount),
                'aux_False.*d_scheme_unbias.*discount_%s.*remark_dppo' % (discount),
            ],
            [
                'PPO undiscounted',
                'PPO discounted',
                'PPO avg',
                'DisPPO undiscounted',
                'DisPPO discounted',
                'DisPPO avg',
            ],
            'ppo_metric_%s' % (discount),
            './log/discounting/aux_ppo',
            [
                'episodic_return_train',
                'discounted_return_train',
                'avg_reward',
                'episodic_return_train',
                'discounted_return_train',
                'avg_reward',
            ],
            dict(ylabel='score', yscale='log')
        ]

    data[44] = [
        [
            'd_scheme_no-discount_1',
            'd_scheme_no-discount_0.995-',
            'd_scheme_no-discount_0.99-',
            'd_scheme_no-discount_0.97',
            'd_scheme_no-discount_0.95',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=0.995$',
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=0.97$',
            r'PPO $\gamma=0.95$',
        ],
        'ppo-ret',
        './log/discounting/no_gae/episodic',
        'episodic_return_train',
    ]

    data[45] = [
        [
            'v_max_20-',
            'v_max_40-',
            'v_max_80-',
            'v_max_160-',
            'v_max_320-',
            'v_max_640-',
        ],
        [
            r'PPO C51 V=20',
            r'PPO C51 V=40',
            r'PPO C51 V=80',
            r'PPO C51 V=160',
            r'PPO C51 V=320',
            r'PPO C51 V=640',
        ],
        'ppo-c51-ret',
        './log/discounting/no_gae/episodic',
        'episodic_return_train',
    ]

    data[46] = [
        [
            'H_16-',
            'H_32-',
            'H_64-',
            'H_128-',
            'H_256-',
        ],
        [
            r'PPO FHTD H=16',
            r'PPO FHTD H=32',
            r'PPO FHTD H=64',
            r'PPO FHTD H=128',
            r'PPO FHTD H=256',
        ],
        'ppo-fhtd-ret',
        './log/discounting/no_gae/episodic',
        'episodic_return_train',
    ]

    data[47] = [
        [
            'd_scheme_no-discount_0.99-',
            'd_scheme_no-discount_1-',
            'v_max_20-',
            'v_max_320-',
            'v_max_640-',
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO C51 $V_{\max}=20$',
            r'PPO C51 $V_{\max}=320$',
            r'PPO C51 $V_{\max}=640$',
        ],
        'ppo-c51-vs-ppo-ret',
        './log/discounting/no_gae/episodic',
        'episodic_return_train',
    ]

    data[48] = [
        [
            'd_scheme_no-discount_0.99-',
            'd_scheme_no-discount_1-',
            'H_16-',
            'H_32-',
            'H_64-',
            'H_128-',
            'H_256-',
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO FHTD H=16',
            r'PPO FHTD H=32',
            r'PPO FHTD H=64',
            r'PPO FHTD H=128',
            r'PPO FHTD H=256',
        ],
        'ppo-fhtd-vs-ppo-ret',
        './log/discounting/no_gae/episodic',
        'episodic_return_train',
    ]

    data[49] = [
        [
            'd_scheme_no-discount_0.99-',
            'aux_False-d_scheme_unbias-discount_0.99-',
            'd_scheme_comp_unbias-discount_0.99-',
            'd_scheme_inv_linear-discount_0.99-',
            'd_scheme_log_linear-discount_0.99-',
        ],
        [
            'PPO',
            r'PPO $\gamma^t$',
            r'PPO $1 - \gamma^{t+1}$',
            r'PPO $1 / (1+t)$',
            r'PPO log(t+1)',
        ],
        'ppo-discounting-scheme-ret',
        './log/discounting/no_gae/episodic',
        'episodic_return_train',
    ]

    for i, discount in enumerate([0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]):
        info = [
            [
                'd_scheme_no-discount_%s-' % (discount),
                'aux_False-d_scheme_unbias-discount_%s-' % (discount),
                # 'aux_True-d_scheme_unbias-discount_%s-.*sync_aux_False' % (discount),
                'aux_True-d_scheme_unbias-discount_%s-.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                'AuxPPO',
                # 'AuxPPO sync',
            ],
            'aux_ppo_%s-dis-ret' % (discount),
            './log/discounting/no_gae/discounted',
            'discounted_return_train',
        ]
        data[50 + i] = info

    data[58] = [
        [
            'd_scheme_no-discount_%s-' % (0.995),
            'aux_False-d_scheme_unbias-discount_%s-' % (0.995),
        ],
        [
            'PPO',
            'DisPPO',
        ],
        'aux_ppo_motivation-dis-ret',
        './log/discounting/no_gae/discounted',
        'discounted_return_train',
    ]

    data[59] = [
        [
            'd_scheme_no-discount_1-.*episodic_return_train',
            'd_scheme_no-discount_0.995-.*episodic_return_train',
            'd_scheme_no-discount_0.99-.*episodic_return_train',
            'd_scheme_no-discount_0.97-.*episodic_return_train',
            'd_scheme_no-discount_0.95-.*episodic_return_train',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=0.995$',
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=0.97$',
            r'PPO $\gamma=0.95$',
        ],
        'ppo-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[60] = [
        [
            ['discount_1-.*v_max_20-', 'aux_True'],
            ['discount_1-.*v_max_40-', 'aux_True'],
            ['discount_1-.*v_max_80-', 'aux_True'],
            ['discount_1-.*v_max_160-', 'aux_True'],
            ['discount_1-.*v_max_320-', 'aux_True'],
            ['discount_1-.*v_max_640-', 'aux_True'],
            ['discount_1-.*v_max_1280-', 'aux_True'],
            ['discount_1-.*v_max_2560-', 'aux_True'],
            ['discount_1-.*v_max_5120-', 'aux_True'],
            ['discount_1-.*v_max_10240-', 'aux_True'],
            ['discount_1-.*v_max_81920-', 'aux_True'],
            ['discount_1-.*v_max_163840-', 'aux_True'],
            ['discount_1-.*v_max_327680-', 'aux_True'],
        ],
        [
            r'PPO C51 V=20',
            r'PPO C51 V=40',
            r'PPO C51 V=80',
            r'PPO C51 V=160',
            r'PPO C51 V=320',
            r'PPO C51 V=640',
            r'PPO C51 V=1280',
            r'PPO C51 V=2560',
            r'PPO C51 V=5120',
            r'PPO C51 V=10240',
            r'PPO C51 V=81920',
            r'PPO C51 V=163840',
            r'PPO C51 V=327680',
        ],
        'ppo-c51-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[61] = [
        [
            'H_16-',
            'H_32-',
            'H_64-',
            'H_128-',
            'H_256-',
            'H_512-',
            'H_1024-',
        ],
        [
            r'PPO FHTD H=16',
            r'PPO FHTD H=32',
            r'PPO FHTD H=64',
            r'PPO FHTD H=128',
            r'PPO FHTD H=256',
            r'PPO FHTD H=512',
            r'PPO FHTD H=1024',
        ],
        'ppo-fhtd-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[62] = [
        [
            'd_scheme_no-discount_0.99-.*episodic_return_train',
            'd_scheme_no-discount_1-.*episodic_return_train',
            ['v_max_320-', 'aux_True'],
            ['v_max_2560-', 'aux_True'],
            ['v_max_5120-', 'aux_True'],
            ['v_max_81920-', 'aux_True'],
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO C51 $V_{\max}=320$',
            r'PPO C51 $V_{\max}=2560$',
            r'PPO C51 $V_{\max}=5120$',
            r'PPO C51 $V_{\max}=81920$',
        ],
        'ppo-c51-vs-ppo-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[63] = [
        [
            'd_scheme_no-discount_0.99-.*episodic_return_train',
            'd_scheme_no-discount_1-.*episodic_return_train',
            'H_64-',
            'H_128-',
            'H_256-',
            'H_512-',
            'H_1024-',
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO FHTD H=64',
            r'PPO FHTD H=128',
            r'PPO FHTD H=256',
            r'PPO FHTD H=512',
            r'PPO FHTD H=1024',
        ],
        'ppo-fhtd-vs-ppo-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    for i, discount in enumerate([0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]):
        info = [
            [
                'd_scheme_no-discount_%s-.*discounted_return_train' % (discount),
                'aux_False-d_scheme_unbias-discount_%s-' % (discount),
                # 'aux_True-d_scheme_unbias-discount_%s-.*sync_aux_False' % (discount),
                'aux_True-d_scheme_unbias-discount_%s-.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                'AuxPPO',
                # 'AuxPPO sync',
            ],
            'aux_ppo_%s-dis-ret' % (discount),
            './log/discounting/timestamp',
            'discounted_return_train',
        ]
        data[63 + i] = info

    data[71] = [
        [
            'd_scheme_no-discount_1-.*episodic_return_train',
            'aux_True.*v_max_20-',
            'aux_True.*v_max_40-',
            'aux_True.*v_max_80-',
            'aux_True.*v_max_160-',
            'aux_True.*v_max_320-',
            'aux_True.*v_max_640-',
            'aux_True.*v_max_1280-',
            'aux_True.*v_max_2560-',
            'aux_True.*v_max_5120-',
            'aux_True.*v_max_81920-',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO C51 Aux V=20',
            r'PPO C51 Aux V=40',
            r'PPO C51 Aux V=80',
            r'PPO C51 Aux V=160',
            r'PPO C51 Aux V=320',
            r'PPO C51 Aux V=640',
            r'PPO C51 Aux V=1280',
            r'PPO C51 Aux V=2560',
            r'PPO C51 Aux V=5120',
            r'PPO C51 Aux V=81920',
        ],
        'ppo-c51-aux-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[72] = [
        [
            'd_scheme_no-discount_1-.*episodic_return_train',
            'H_16-.*aux_True',
            'H_32-.*aux_True',
            'H_64-.*aux_True',
            'H_128-.*aux_True',
            'H_256-.*aux_True',
            'H_512-.*aux_True',
            'H_1024-.*aux_True',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO FHTD Aux H=16',
            r'PPO FHTD Aux H=32',
            r'PPO FHTD Aux H=64',
            r'PPO FHTD Aux H=128',
            r'PPO FHTD Aux H=256',
            r'PPO FHTD Aux H=512',
            r'PPO FHTD Aux H=1024',
        ],
        'ppo-fhtd-aux-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    for i, discount in enumerate([0.95, 0.97, 0.99, 0.995]):
        data[73 + i] = [
            [
                ['d_scheme_no-discount_1-.*episodic_return_train', '(adv_gamma|extra_data)'],
                ['d_scheme_no-discount_%s-.*episodic_return_train' % (discount), '(adv_gamma|extra_data)'],
                'adv_gamma_1-.*d_scheme_no-discount_%s-.*episodic_return_train' % (discount),
            ],
            [
                r'PPO $\gamma=1$',
                r'PPO $\gamma=%s$' % (discount),
                r'PPO $\gamma_A=1, \gamma=%s$' % (discount),
            ],
            'ppo-adv_gamma-ret_%s' % (discount),
            './log/discounting/timestamp',
            'episodic_return_train',
        ]

    data[77] = [
        [
            ['d_scheme_no-discount_1-.*episodic_return_train', '(adv_gamma|extra_data)'],
            ['d_scheme_no-discount_0.99-.*episodic_return_train', '(adv_gamma|extra_data)'],
            'd_scheme_no-discount_1-extra_data_2.*episodic_return_train',
            'd_scheme_no-discount_1-extra_data_4.*episodic_return_train',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=0.99$',
            r'PPO $2 \times$ data, $\gamma=1$',
            r'PPO $4 \times$ data, $\gamma=1$',
        ],
        'ppo-extra_data-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    for i, discount in enumerate([0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]):
        info = [
            [
                'd_scheme_no-discount_%s-flip_r.*discounted_return_train' % (discount),
                'aux_False-d_scheme_unbias-discount_%s-flip_r' % (discount),
                'aux_True-d_scheme_unbias-discount_%s-flip_r.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                'AuxPPO',
            ],
            'flip_r_aux_ppo_%s-dis-ret' % (discount),
            './log/discounting/timestamp',
            'discounted_return_train',
        ]
        data[77 + i] = info

    for i, discount in enumerate([0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]):
        info = [
            [
                'd_scheme_no-discount_%s-flip_r.*discounted_return_train' % (discount),
                'aux_False-d_scheme_unbias-discount_%s-flip_r' % (discount),
                'aux_True-d_scheme_unbias-discount_%s-flip_r.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                'AuxPPO',
            ],
            'episode_len_flip_r_aux_ppo_%s' % (discount),
            './log/discounting/timestamp',
            'episode_len',
        ]
        data[85 + i] = info

    for i, discount in enumerate([0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]):
        info = [
            [
                ['d_scheme_no-discount_%s-.*discounted_return_train' % (discount), 'flip_r'],
                ['aux_False-d_scheme_unbias-discount_%s-' % (discount), 'flip_r'],
                ['aux_True-d_scheme_unbias-discount_%s-.*sync_aux_True' % (discount), 'flip_r'],
            ],
            [
                'PPO',
                'DisPPO',
                'AuxPPO',
            ],
            'episode_len_aux_ppo_%s' % (discount),
            './log/discounting/timestamp',
            'episode_len',
        ]
        data[93 + i] = info

    data[101] = [
        [
            'critic_update_td-d_scheme_no-discount_1-.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_0.995-.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_0.99-.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_0.97-.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_0.95-.*episodic_return_train',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=0.995$',
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=0.97$',
            r'PPO $\gamma=0.95$',
        ],
        'ppo-td-critic-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[102] = [
        [
            'critic_update_td-d_scheme_no-discount_0.99-.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_1-.*episodic_return_train',
            ['v_max_320-', 'aux_True'],
            ['v_max_2560-', 'aux_True'],
            ['v_max_5120-', 'aux_True'],
            ['v_max_81920-', 'aux_True'],
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO C51 $V_{\max}=320$',
            r'PPO C51 $V_{\max}=2560$',
            r'PPO C51 $V_{\max}=5120$',
            r'PPO C51 $V_{\max}=81920$',
        ],
        'ppo-c51-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[103] = [
        [
            'critic_update_td-d_scheme_no-discount_0.99-.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_1-.*episodic_return_train',
            ['H_64-', 'aux_True'],
            ['H_128-', 'aux_True'],
            ['H_256-', 'aux_True'],
            ['H_512-', 'aux_True'],
            ['H_1024-', 'aux_True'],
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO FHTD H=64',
            r'PPO FHTD H=128',
            r'PPO FHTD H=256',
            r'PPO FHTD H=512',
            r'PPO FHTD H=1024',
        ],
        'ppo-fhtd-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    for i, discount in enumerate([0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]):
        info = [
            [
                'd_scheme_no-discount_%s-flip_r.*discounted_return_train' % (discount),
                'aux_False-d_scheme_unbias-discount_%s-flip_r' % (discount),
                'aux_True-d_scheme_unbias-discount_%s-flip_r.*sync_aux_True' % (discount),
            ],
            [
                'PPO',
                'DisPPO',
                'AuxPPO',
            ],
            'Ant-flip_r_aux_ppo_%s-dis-ret' % (discount),
            './log/discounting/timestamp',
            'discounted_return_train',
        ]
        data[103 + i] = info

    for i, discount in enumerate([0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]):
        info = [
            [
                ['d_scheme_no-discount_%s-.*discounted_return_train' % (discount), 'flip_r'],
                ['aux_False-d_scheme_unbias-discount_%s-' % (discount), 'flip_r'],
                ['aux_True-d_scheme_unbias-discount_%s-.*sync_aux_True' % (discount), 'flip_r'],
            ],
            [
                'PPO',
                'DisPPO',
                'AuxPPO',
            ],
            'Ant-aux_ppo_%s' % (discount),
            './log/discounting/timestamp',
            'discounted_return_train',
        ]
        data[111 + i] = info

    data[119] = [
        [
            ['d_scheme_no-discount_%s-.*discounted_return_train' % (0.995), 'flip_r'],
            ['aux_False-d_scheme_unbias-discount_%s-' % (0.995), 'flip_r'],
        ],
        [
            'PPO',
            'DisPPO',
        ],
        'aux_ppo_motivation-dis-ret',
        './log/discounting/timestamp',
        'discounted_return_train',
    ]

    data[120] = [
        [
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_0.99-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_1-extra_data_2.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_1-extra_data_4.*episodic_return_train',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=0.99$',
            r'PPO $2 \times$ data, $\gamma=1$',
            r'PPO $4 \times$ data, $\gamma=1$',
        ],
        'ppo-td-critic-extra_data-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[121] = [
        [
            'critic_update_td-d_scheme_no-discount_0.99-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            ['discount_0.99-.*v_max_320-', 'aux_True'],
            ['discount_0.99-.*v_max_2560-', 'aux_True'],
            ['discount_0.99-.*v_max_5120-', 'aux_True'],
            ['discount_0.99-.*v_max_81920-', 'aux_True'],
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO C51 $V_{\max}=320, \gamma=0.99$',
            r'PPO C51 $V_{\max}=2560, \gamma=0.99$',
            r'PPO C51 $V_{\max}=5120, \gamma=0.99$',
            r'PPO C51 $V_{\max}=81920, \gamma=0.99$',
        ],
        'ppo-discount-c51-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[122] = [
        [
            'critic_update_td-d_scheme_no-discount_0.99-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            ['H_64-.*discount_0.99', '(aux_True|active)'],
            ['H_128-.*discount_0.99', '(aux_True|active)'],
            ['H_256-.*discount_0.99', '(aux_True|active)'],
            ['H_512-.*discount_0.99', '(aux_True|active)'],
            ['H_1024-.*discount_0.99', '(aux_True|active)'],
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO FHTD H=64, $\gamma=0.99$',
            r'PPO FHTD H=128, $\gamma=0.99$',
            r'PPO FHTD H=256, $\gamma=0.99$',
            r'PPO FHTD H=512, $\gamma=0.99$',
            r'PPO FHTD H=1024, $\gamma=0.99$',
        ],
        'ppo-discount-fhtd-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[123] = [
        [
            'critic_update_td-d_scheme_no-discount_0.99-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            'active_63-.*discount_1',
            'active_127-.*discount_1',
            'active_255-.*discount_1',
            'active_511-.*discount_1',
            'active_1023-.*discount_1',
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=1$',
            r'PPO FHTD active H=64',
            r'PPO FHTD active H=128',
            r'PPO FHTD active H=256',
            r'PPO FHTD active H=512',
            r'PPO FHTD active H=1024',
        ],
        'ppo-active-fhtd-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    for i, v_max in enumerate([20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 81920, 163840, 327680]):
        data[124 + i] = [
            [
                ['discount_0.99-.*v_max_%s-' % (v_max), 'aux_True'],
                ['discount_1-.*v_max_%s-' % (v_max), 'aux_True'],
            ],
            [
                r'PPO C51 $V_{\max}=%s, \gamma=0.99$' % (v_max),
                r'PPO C51 $V_{\max}=%s, \gamma=1$' % (v_max),
            ],
            'ppo-discount-c51-%s-ret' % (v_max),
            './log/discounting/timestamp',
            'episodic_return_train',
        ]

    data[139] = [
        [
            ['discount_1-.*v_max_20-', 'aux_True'],
            ['discount_1-.*v_max_40-', 'aux_True'],
            ['discount_1-.*v_max_80-', 'aux_True'],
            ['discount_1-.*v_max_160-', 'aux_True'],
            ['discount_1-.*v_max_320-', 'aux_True'],
            ['discount_1-.*v_max_640-', 'aux_True'],
            ['discount_1-.*v_max_1280-', 'aux_True'],
            ['discount_1-.*v_max_2560-', 'aux_True'],
            ['discount_1-.*v_max_5120-', 'aux_True'],
            ['discount_1-.*v_max_10240-', 'aux_True'],
            ['discount_1-.*v_max_81920-', 'aux_True'],
            ['discount_1-.*v_max_163840-', 'aux_True'],
            ['discount_1-.*v_max_327680-', 'aux_True'],
        ],
        [
            r'PPO C51 V=20',
            r'PPO C51 V=40',
            r'PPO C51 V=80',
            r'PPO C51 V=160',
            r'PPO C51 V=320',
            r'PPO C51 V=640',
            r'PPO C51 V=1280',
            r'PPO C51 V=2560',
            r'PPO C51 V=5120',
            r'PPO C51 V=10240',
            r'PPO C51 V=81920',
            r'PPO C51 V=163840',
            r'PPO C51 V=327680',
        ],
        'ppo-c51-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[140] = [
        [
            'critic_update_td-d_scheme_no-discount_0.99-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_0.995-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            ['discount_1-.*v_max_320-', 'aux_True'],
            ['discount_1-.*v_max_1280-', 'aux_True'],
            ['discount_1-.*v_max_2560-', 'aux_True'],
            ['discount_1-.*v_max_163840-', 'aux_True'],
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=0.995$',
            r'PPO $\gamma=1$',
            r'PPO C51 $V_{\max}=320$',
            r'PPO C51 $V_{\max}=1280$',
            r'PPO C51 $V_{\max}=2560$',
            r'PPO C51 $V_{\max}=163840$',
        ],
        'ppo-c51-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[141] = [
        [
            # 'critic_update_td-d_scheme_no-discount_0.99-extra_data_0.*episodic_return_train',
            # 'critic_update_td-d_scheme_no-discount_0.995-extra_data_0.*episodic_return_train',
            # 'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            ['discount_1-.*v_max_320-', 'aux_True'],
            ['discount_0.99-.*v_max_320-', 'aux_True'],
            ['discount_1-.*v_max_1280-', 'aux_True'],
            ['discount_0.99-.*v_max_1280-', 'aux_True'],
            ['discount_1-.*v_max_2560-', 'aux_True'],
            ['discount_0.99-.*v_max_2560-', 'aux_True'],
            ['discount_1-.*v_max_163840-', 'aux_True'],
            ['discount_0.99-.*v_max_163840-', 'aux_True'],
            ['discount_1-.*v_max_327680-', 'aux_True'],
            ['discount_0.99-.*v_max_327680-', 'aux_True'],
        ],
        [
            r'PPO C51 $V_{\max}=320$',
            r'PPO C51 $V_{\max}=320, \gamma=0.99$',
            r'PPO C51 $V_{\max}=1280$',
            r'PPO C51 $V_{\max}=1280, \gamma=0.99$',
            r'PPO C51 $V_{\max}=2560$',
            r'PPO C51 $V_{\max}=2560, \gamma=0.99$',
            r'PPO C51 $V_{\max}=163840$',
            r'PPO C51 $V_{\max}=163840, \gamma=0.99$',
            r'PPO C51 $V_{\max}=327680$',
            r'PPO C51 $V_{\max}=327680, \gamma=0.99$',
        ],
        'ppo-discount-c51-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    with open('./deep_rl/data_plot/best_v_max.pkl', 'rb') as f:
        best_v_max = pickle.load(f)

    def patterns_fn(game):
        v_max = best_v_max[game]
        patterns = [
            ['discount_1-.*v_max_%s' % (v_max), 'aux_True'],
            ['discount_0.99-.*v_max_%s' % (v_max), 'aux_True'],
            ['discount_0.995-.*v_max_%s' % (v_max), 'aux_True'],
        ]
        labels = [
            r'PPO C51 $V_{\max}=%s, \gamma=1$' % (v_max),
            r'PPO C51 $V_{\max}=%s, \gamma=0.99$' % (v_max),
            r'PPO C51 $V_{\max}=%s, \gamma=0.995$' % (v_max),
        ]
        return patterns, labels

    data[142] = [
        [
            ['discount_1-.*v_max_320-', 'aux_True'],
            ['discount_1-.*v_max_1280-', 'aux_True'],
            ['discount_1-.*v_max_2560-', 'aux_True'],
            ['discount_1-.*v_max_163840-', 'aux_True'],
        ],
        [
            r'PPO $\gamma=0.99$',
            r'PPO $\gamma=0.995$',
            r'PPO $\gamma=1$',
            r'PPO C51 $V_{\max}=320$',
            r'PPO C51 $V_{\max}=1280$',
            r'PPO C51 $V_{\max}=2560$',
            r'PPO C51 $V_{\max}=163840$',
        ],
        'ppo-c51-best-v-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
        {},
        patterns_fn,
    ]

    for i, discount in enumerate([0.95, 0.97, 0.99, 0.995]):
        data[143 + i] = [
            [
                'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
                'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
                'adv_gamma_1-.*critic_update_td.*d_scheme_no-discount_%s-.*episodic_return_train' % (discount),
            ],
            [
                r'PPO $\gamma_A=1, \gamma_C=1$',
                r'PPO $\gamma_A=%s, \gamma_C=%s$' % (discount, discount),
                r'PPO $\gamma_A=1, \gamma_C=%s$' % (discount),
            ],
            'ppo-td-adv_gamma-ret_%s' % (discount),
            './log/discounting/timestamp',
            'episodic_return_train',
        ]

    for i, discount in enumerate([0.99, 0.995, 1]):
        data[147 + i] = [
            [
                'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
                'critic_update_td-d_scheme_no-discount_%s-extra_data_2.*episodic_return_train' % (discount),
                'critic_update_td-d_scheme_no-discount_%s-extra_data_4.*episodic_return_train' % (discount),
            ],
            [
                r'PPO $\gamma=%s$' % (discount),
                r'PPO $2 \times$ data, $\gamma=%s$' % (discount),
                r'PPO $4 \times$ data, $\gamma=%s$' % (discount),
            ],
            'ppo-td-critic-%s-extra_data-ret' % (discount),
            './log/discounting/timestamp',
            'episodic_return_train',
        ]

    data[150] = [
        [
            ['d_scheme_no-discount_1.*episodic_return_train', '(critic_update_td|extra_data|aux_True|adv_gamma)'],
            ['d_scheme_no-discount_0.995-.*episodic_return_train', '(critic_update_td|extra_data|aux_True|adv_gamma)'],
            ['d_scheme_no-discount_0.99-.*episodic_return_train', '(critic_update_td|extra_data|aux_True|adv_gamma)'],
            # 'd_scheme_no-discount_0.97',
            # 'd_scheme_no-discount_0.95',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=0.995$',
            r'PPO $\gamma=0.99$',
            # r'PPO $\gamma=0.97$',
            # r'PPO $\gamma=0.95$',
        ],
        'ppo-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[151] = [
        [
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_0.995-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_0.99-extra_data_0.*episodic_return_train',
        ],
        [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=0.995$',
            r'PPO $\gamma=0.99$',
        ],
        'ppo-td-critic-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    data[152] = [
        [
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            ['d_scheme_no-discount_1.*episodic_return_train', '(critic_update_td|extra_data|aux_True|adv_gamma)'],
        ],
        [
            r'PPO TD $\gamma=1$',
            r'PPO MC $\gamma=1$',
        ],
        'ppo-td-mc-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
    ]

    def patterns_fn(game):
        if game in ['Hopper-v2', 'Humanoid-v2', 'Ant-v2']:
            discount = 0.995
        else:
            discount = 0.99
        patterns = [
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
            ['H_64-discount_1-', '(aux_True|active)'],
            ['H_128-discount_1-', '(aux_True|active)'],
            ['H_256-discount_1-', '(aux_True|active)'],
            ['H_512-discount_1-', '(aux_True|active)'],
            ['H_1024-discount_1-', '(aux_True|active)'],
        ]
        labels = [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=%s$' % (discount),
            r'PPO FHTD H=64',
            r'PPO FHTD H=128',
            r'PPO FHTD H=256',
            r'PPO FHTD H=512',
            r'PPO FHTD H=1024',
        ]
        return patterns, labels

    data[153] = [
        [None] * 7,
        [None] * 7,
        'ppo-fhtd-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
        {},
        patterns_fn,
    ]

    def patterns_fn(game):
        if game in ['Hopper-v2', 'Humanoid-v2', 'Ant-v2']:
            discount = 0.995
        else:
            discount = 0.99
        patterns = [
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
            'active_63-.*discount_1',
            'active_127-.*discount_1',
            'active_255-.*discount_1',
            'active_511-.*discount_1',
            'active_1023-.*discount_1',
        ]
        labels = [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=%s$' % (discount),
            r'PPO FHTD H=64',
            r'PPO FHTD H=128',
            r'PPO FHTD H=256',
            r'PPO FHTD H=512',
            r'PPO FHTD H=1024',
        ]
        return patterns, labels

    data[154] = [
        [None] * 7,
        [None] * 7,
        'ppo-active-fhtd-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
        {},
        patterns_fn,
    ]

    def patterns_fn(game):
        v_max = best_v_max[game]
        if game in ['Hopper-v2', 'Humanoid-v2', 'Ant-v2']:
            discount = 0.995
        else:
            discount = 0.99
        patterns = [
            'critic_update_td-d_scheme_no-discount_1-extra_data_0.*episodic_return_train',
            'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
            ['discount_1-.*v_max_%s-' % (v_max), 'aux_True'],
        ]
        labels = [
            r'PPO $\gamma=1$',
            r'PPO $\gamma=%s$' % (discount),
            r'PPO C51 $V_{\max} = %s$' % (v_max),
        ]
        return patterns, labels

    data[154] = [
        [None] * 3,
        [None] * 3,
        'ppo-c51-vs-ppo-td-ret',
        './log/discounting/timestamp',
        'episodic_return_train',
        {},
        patterns_fn
    ]

    def patterns_fn(game):
        with open('./deep_rl/data_plot/best_v_max_0.99.pkl', 'rb') as f:
            best_v_max = pickle.load(f)
        v_max = best_v_max[game]
        # if game in ['Hopper-v2', 'Humanoid-v2', 'Ant-v2']:
        #     discount = 0.995
        # else:
        discount = 0.99
        patterns = [
            'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
            ['discount_0.99-.*v_max_%s-' % (v_max), 'aux_True'],
        ]
        labels = [
            r'PPO $\gamma=%s$' % (discount),
            r'PPO C51 $V_{\max} = %s$' % (v_max),
        ]
        return patterns, labels

    data[155] = [
        [None] * 2,
        [None] * 2,
        'ppo-c51-vs-ppo-td-ret-0.99',
        './log/discounting/timestamp',
        'episodic_return_train',
        {},
        patterns_fn
    ]

    def patterns_fn(game):
        with open('./deep_rl/data_plot/best_v_max.pkl', 'rb') as f:
            best_v_max = pickle.load(f)
        v_max = best_v_max[game]
        discount = 1
        patterns = [
            'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
            ['discount_1-.*v_max_%s-' % (v_max), 'aux_True'],
        ]
        labels = [
            r'PPO $\gamma=%s$' % (discount),
            r'PPO C51 $V_{\max} = %s, \gamma=1$' % (v_max),
        ]
        return patterns, labels

    data[156] = [
        [None] * 2,
        [None] * 2,
        'ppo-c51-vs-ppo-td-ret-1',
        './log/discounting/timestamp',
        'episodic_return_train',
        {},
        patterns_fn
    ]

    def patterns_fn(game):
        with open('./deep_rl/data_plot/best_v_max_0.995.pkl', 'rb') as f:
            best_v_max = pickle.load(f)
        v_max = best_v_max[game]
        discount = 0.995
        patterns = [
            'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
            ['discount_0.995-.*v_max_%s-' % (v_max), 'aux_True'],
        ]
        labels = [
            r'PPO $\gamma=%s$' % (discount),
            r'PPO C51 $V_{\max} = %s$' % (v_max),
        ]
        return patterns, labels

    data[157] = [
        [None] * 2,
        [None] * 2,
        'ppo-c51-vs-ppo-td-ret-0.995',
        './log/discounting/timestamp',
        'episodic_return_train',
        {},
        patterns_fn
    ]

    def patterns_fn(game):
        if game in ['Hopper-v2', 'Humanoid-v2', 'Ant-v2']:
            discount = 0.995
        else:
            discount = 0.99
        with open('./deep_rl/data_plot/best_v_max_%s.pkl' % (discount), 'rb') as f:
            best_v_max = pickle.load(f)
        v_max = best_v_max[game]

        patterns = [
            'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
            ['discount_%s-.*v_max_%s-' % (discount, v_max), 'aux_True'],
        ]
        labels = [
            r'PPO $\gamma=%s$' % (discount),
            r'PPO C51 $V_{\max} = %s, \gamma=%s$' % (v_max, discount),
        ]
        return patterns, labels

    data[158] = [
        [None] * 2,
        [None] * 2,
        'ppo-c51-vs-ppo-td-ret-dis',
        './log/discounting/timestamp',
        'episodic_return_train',
        {},
        patterns_fn
    ]

    for i, discount in enumerate([0.99, 0.995, 1]):
        data[159 + i] = [
            [
                'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
                'critic_update_td-d_scheme_no-discount_%s-gae_tau_0-.*-lr_type_episodic_return_train-multi_path_1' % (
                    discount),
                'critic_update_td-d_scheme_no-discount_%s-gae_tau_0-.*-lr_type_episodic_return_train-multi_path_2' % (
                    discount),
                'critic_update_td-d_scheme_no-discount_%s-gae_tau_0-.*-lr_type_episodic_return_train-multi_path_4' % (
                    discount),
            ],
            [
                r'PPO MP=0 $\gamma=%s$' % (discount),
                r'PPO MP=1 $\gamma=%s$' % (discount),
                r'PPO MP=2 $\gamma=%s$' % (discount),
                r'PPO MP=4 $\gamma=%s$' % (discount),
            ],
            'ppo-td-mp-ret-%s' % (discount),
            './log/discounting/timestamp',
            'episodic_return_train',
        ]

    for i, discount in enumerate([0.99, 0.995, 1]):
        data[162 + i] = [
            [
                'critic_update_td-d_scheme_no-discount_%s-extra_data_0.*episodic_return_train' % (discount),
                'critic_update_td-discount_%s-gae_tau_0-.*-lr_type_episodic_return_train-num_quantiles_25' % (discount),
                'critic_update_td-discount_%s-gae_tau_0-.*-lr_type_episodic_return_train-num_quantiles_50' % (
                    discount),
                'critic_update_td-discount_%s-gae_tau_0-.*-lr_type_episodic_return_train-num_quantiles_100' % (
                    discount),
            ],
            [
                r'PPO  $\gamma=%s$' % (discount),
                r'PPO QR 25 $\gamma=%s$' % (discount),
                r'PPO QR 50 $\gamma=%s$' % (discount),
                r'PPO QR 100 $\gamma=%s$' % (discount),
            ],
            'ppo-td-qr-ret-%s' % (discount),
            './log/discounting/timestamp',
            'episodic_return_train',
        ]

    lr_patterns = []
    for lr_a in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        for r in [1, 3]:
            lr_patterns.append('(%s, %s)' % (lr_a, r))
    data[170] = [
        ['lr_%d' % lr for lr in range(10)],
        lr_patterns,
        'lr_search_group1',
        './log/discounting/rebuttal/lr_search_group1',
    ]

    data[171] = [
        [
            'discount_0.99.*use_target_net_True',
            'discount_0.99.*use_target_net_False',
        ],
        [
            'w/ target net',
            'w/o target net',
        ],
        'target_net_discount_0.99',
        './log/discounting/icml_rebuttal/ppo44',
        'episodic_return_train',
    ]

    data[172] = [
        [
            'discount_1.*use_target_net_True',
            'discount_1.*use_target_net_False',
        ],
        [
            'w/ target net',
            'w/o target net',
        ],
        'target_net_discount_1',
        './log/discounting/icml_rebuttal/ppo44',
        'episodic_return_train',
    ]

    ########## NeurIPS #####################

    data[173] = [
        [
            'critic_update_mc.*d_scheme_no.*discount_1-',
            'critic_update_mc.*d_scheme_no.*discount_0.995-',
            'critic_update_mc.*d_scheme_no.*discount_0.99-',
            'critic_update_mc.*d_scheme_no.*discount_0.97-',
            'critic_update_mc.*d_scheme_no.*discount_0.95-',
        ],
        [
            r'$\gamma_C = 1$',
            r'$\gamma_C = 0.995$',
            r'$\gamma_C = 0.99$',
            r'$\gamma_C = 0.97$',
            r'$\gamma_C = 0.95$',
        ],
        'figure1',
        './log/discounting/neurips',
        'episodic_return_train',
    ]

    data[174] = [
        [
            'critic_update_mc.*d_scheme_no.*discount_1-',
            'critic_update_td.*d_scheme_no.*discount_1-.*remark_ppo_td-',
        ],
        [
            'PPO',
            'PPO-TD',
        ],
        'figure2',
        './log/discounting/neurips',
        'episodic_return_train',
    ]

    data[175] = [
        [
            'critic_update_td.*d_scheme_no.*discount_1-.*remark_ppo_td-',
            'critic_update_td.*d_scheme_no.*discount_0.995-.*remark_ppo_td-',
            'critic_update_td.*d_scheme_no.*discount_0.99-.*remark_ppo_td-',
            'critic_update_td.*d_scheme_no.*discount_0.97-.*remark_ppo_td-',
            'critic_update_td.*d_scheme_no.*discount_0.95-.*remark_ppo_td-',
        ],
        [
            r'$\gamma_C = 1$',
            r'$\gamma_C = 0.995$',
            r'$\gamma_C = 0.99$',
            r'$\gamma_C = 0.97$',
            r'$\gamma_C = 0.95$',
        ],
        'figure3',
        './log/discounting/neurips',
        'episodic_return_train',
    ]

    for i, gamma in enumerate([0.99, 0.995, 1]):
        data[176 + i] = [
            [
                f'critic_update_td.*d_scheme_no.*discount_{gamma}-.*multi_path_0.*remark_ppo_td_ex',
                f'critic_update_td.*d_scheme_no.*discount_{gamma}-.*multi_path_1.*remark_ppo_td_ex',
                f'critic_update_td.*d_scheme_no.*discount_{gamma}-.*multi_path_2.*remark_ppo_td_ex',
                f'critic_update_td.*d_scheme_no.*discount_{gamma}-.*multi_path_4.*remark_ppo_td_ex',
            ],
            [
                r'$N = 0$',
                r'$N = 1$',
                r'$N = 2$',
                r'$N = 4$',
            ],
            f'figure{4 + i}',
            './log/discounting/neurips',
            'episodic_return_train',
        ]

    Hs = [16, 32, 64, 128, 256, 512, 1024]
    data[179] = [
        [f'H_{H}-.*ppo_fhtd1' for H in Hs[:-1]] + ['active_1023.*ppo_fhtd2'] + data[175][0],
        Hs + data[175][1],
        'figure6',
        './log/discounting/neurips',
        'episodic_return_train',
    ]

    data[180] = [
        [f'active_{H - 1}-.*ppo_fhtd2' for H in Hs],
        Hs,
        'figure7',
        './log/discounting/neurips',
        'episodic_return_train',
    ]

    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]
    best_gamma = [0.99, 0.99, 0.995, 0.97, 0.99, 0.95]
    best_H = [128, 256, 512, 64, 512, 32]
    best_gamma = {game: gamma for game, gamma in zip(games, best_gamma)}
    best_H = {game: H for game, H in zip(games, best_H)}
    # def patterns_fn(game):
    #     gamma = best_gamma[game]
    #     H = best

    for i, gamma in enumerate([0.95, 0.97, 0.99, 0.995]):
        data[181 + i] = [
            [
                f'critic_update_mc.*d_scheme_no.*discount_{gamma}-.*lr_type_discounted_return_train',
                f'critic_update_mc.*d_scheme_unbias.*discount_{gamma}-.*lr_type_discounted_return_train',
            ],
            [
                r'PPO $\gamma_C = %s$' % (gamma),
                r'DisPPO $\gamma_C = %s$' % (gamma),
            ],
            f'figure10_{gamma}',
            './log/discounting/neurips',
            'discounted_return_train',

        ]

    p = Pool(10)
    p.starmap(plot_ppo_patterns, [data[i] for i in range(181, 185)])
    # plot_ppo_patterns(*data[181])


def lr_search(dir=None, filename=None, root=None, out_dir=None):
    plotter = Plotter()
    if root is None:
        root = './log/discounting/lr_search/timestamp'
    if out_dir is None:
        out_dir = 'data'
    root = os.path.join(root, dir)
    info = dict()
    tags = ['discounted_return_train', 'episodic_return_train']

    fields = ['lr_%d' % lr for lr in range(10)]

    def score_fn(ys):
        return np.mean([y[-100:] for y in ys])

    for tag in tags:
        info[tag] = plotter.reduce(root, tag, fields, score_fn)
    with open('deep_rl/%s/%s.pkl' % (out_dir, filename), 'wb') as f:
        pickle.dump(info, f)


def lr_search_view(filename):
    with open('deep_rl/data/%s.pkl' % (filename), 'rb') as f:
        info = pickle.load(f)
    infos = []
    for key in info.keys():
        infos.append(info[key])
    for key in infos[0].keys():
        print(key, infos[0][key], infos[1][key])


def compute_best_v_max(discount):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]
    vs = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 81920, 163840, 327680]

    def score_fn(ys):
        return np.mean([y[-100:] for y in ys])

    best_v = {}
    plotter = Plotter()
    for game in games:
        patterns = ['.*%s.*discount_%s-.*v_max_%s-' % (game, discount, v) for v in vs]
        scores = []
        for p in patterns:
            log_dirs = plotter.filter_log_dirs(pattern=p, negative_pattern='.*aux_True.*',
                                               root='./log/discounting/timestamp')
            xy_list = plotter.load_log_dirs(log_dirs, tag='episodic_return_train')
            scores.append(score_fn([y for x, y in xy_list]))
        best_v[game] = vs[np.nanargmax(scores)]
    print(best_v)
    with open('./deep_rl/data_plot/best_v_max_%s.pkl' % (discount), 'wb') as f:
        pickle.dump(best_v, f)


# FIGURE_TYPE = 'small'
FIGURE_TYPE = 'large'
def plot_paper_impl(patterns,
                    labels,
                    filename,
                    dir,
                    field='episodic_return_train',
                    patterns_fn=None,
                    legends=False,
                    legend_size=None,
                    legend_loc=None,
                    params={}):
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]
    if labels is None:
        labels = patterns

    if isinstance(field, list):
        fields = field
    else:
        fields = [field] * len(patterns)

    patterns = [[p, ' '] if isinstance(p, str) else p for p in patterns]

    fontsize = 20
    if legend_size is None:
        legend_size = 16

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        if FIGURE_TYPE == 'small':
            plt.figure(figsize=(l * 5, 5))
        elif FIGURE_TYPE == 'large':
            plt.figure(figsize=(3 * 5, 2 * 5))
        else:
            raise NotImplementedError
        for i, game in enumerate(games):
            if FIGURE_TYPE == 'small':
                plt.subplot(1, l, i + 1)
            elif FIGURE_TYPE == 'large':
                plt.subplot(2, 3, i + 1)
            else:
                raise NotImplementedError
            if patterns_fn is None:
                patterns = kwargs['patterns']
                labels = kwargs['labels']
            else:
                patterns, labels = patterns_fn(game)
            for j, p in enumerate(patterns):
                label = labels[j]
                color = self.COLORS[j]
                positve_p, negative_p = (p, ' ') if isinstance(p, str) else p
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, positve_p),
                                                negative_pattern='.*%s.*' % (negative_p),
                                                **kwargs)
                kwargs['tag'] = fields[j]
                x, y = self.load_results(log_dirs, **kwargs)
                assert y.shape[0] == 30 or y.shape[0] == 3
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                assert y.shape[0] == 30 or y.shape[0] == 3
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
            if i == 0:
                plt.ylabel(kwargs['ylabel'], fontsize=fontsize)
            plt.xlabel('steps', fontsize=fontsize)
            plt.xticks([0, 2e6], [0, '2M'], fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)
            plt.yscale(kwargs['yscale'])
            plt.title(game[:-3], fontsize=fontsize)
            if legends or i == 0:
                plt.legend(fontsize=legend_size, loc=legend_loc)

    params.setdefault('downsample', 100)
    params.setdefault('window', 10)
    params.setdefault('interpolation', 100)
    params.setdefault('ylabel', 'undiscounted return')
    params.setdefault('yscale', 'linear')
    plot_games(plotter,
               games=games,
               patterns=patterns,
               agg='mean',
               labels=labels,
               right_align=False,
               tag=field,
               root=dir,
               **params,
               )

    # plt.show()
    plt.tight_layout()
    if FIGURE_TYPE == 'small':
        filename = '/Users/Shangtong/GoogleDrive/Paper/discounting/img/%s.pdf' % (filename)
    elif FIGURE_TYPE == 'large':
        filename = '/Users/Shangtong/GoogleDrive/Paper/discounting/img/%s_large.pdf' % (filename)
    else:
        raise NotImplementedError
    plt.savefig(filename, bbox_inches='tight')


def plot_paper():
    data = {}

    data[0] = [
        [
            'critic_update_mc.*d_scheme_no.*discount_1-.*episodic_return',
            'critic_update_mc.*d_scheme_no.*discount_0.995-.*episodic_return',
            'critic_update_mc.*d_scheme_no.*discount_0.99-.*episodic_return',
            'critic_update_mc.*d_scheme_no.*discount_0.97-.*episodic_return',
            'critic_update_mc.*d_scheme_no.*discount_0.95-.*episodic_return',
        ],
        [
            r'PPO $\gamma_C=1$',
            r'PPO $\gamma_C=0.995$',
            r'PPO $\gamma_C=0.99$',
            r'PPO $\gamma_C=0.97$',
            r'PPO $\gamma_C=0.95$',
        ],
        'ppo',
        './log/discounting/neurips',
        'episodic_return_train',
        None, False, None, 'lower left'
    ]

    data[1] = [
        [
            'critic_update_mc.*d_scheme_no.*discount_1-.*episodic_return',
            'critic_update_td.*d_scheme_no.*discount_1-.*remark_ppo_td-',
        ],
        [
            r'PPO $\gamma_C=1$',
            r'PPO-TD $\gamma_C=1$',
        ],
        'ppo-vs-ppo-td',
        './log/discounting/neurips',
        'episodic_return_train',
    ]

    data[2] = [
        [
            'critic_update_td.*d_scheme_no.*discount_1-.*remark_ppo_td-',
            'critic_update_td.*d_scheme_no.*discount_0.995-.*remark_ppo_td-',
            'critic_update_td.*d_scheme_no.*discount_0.99-.*remark_ppo_td-',
            'critic_update_td.*d_scheme_no.*discount_0.97-.*remark_ppo_td-',
            'critic_update_td.*d_scheme_no.*discount_0.95-.*remark_ppo_td-',
        ],
        [
            r'PPO-TD $\gamma_C=1$',
            r'PPO-TD $\gamma_C=0.995$',
            r'PPO-TD $\gamma_C=0.99$',
            r'PPO-TD $\gamma_C=0.97$',
            r'PPO-TD $\gamma_C=0.95$',
        ],
        'ppo-td',
        './log/discounting/neurips',
        'episodic_return_train',
        None, False, None, 'lower left'
    ]

    for i, discount in enumerate([0.99, 0.995, 1]):
        data[3 + i] = [
            [
                f'critic_update_td.*d_scheme_no.*discount_{discount}-.*multi_path_0.*remark_ppo_td_ex',
                f'critic_update_td.*d_scheme_no.*discount_{discount}-.*multi_path_1.*remark_ppo_td_ex',
                f'critic_update_td.*d_scheme_no.*discount_{discount}-.*multi_path_2.*remark_ppo_td_ex',
                f'critic_update_td.*d_scheme_no.*discount_{discount}-.*multi_path_4.*remark_ppo_td_ex',

            ],
            [
                r'PPO-TD-Ex N=0',
                r'PPO-TD-Ex N=1',
                r'PPO-TD-Ex N=2',
                r'PPO-TD-Ex N=4',
            ],
            'ppo-td-ex-%s' % (discount),
            './log/discounting/neurips',
            'episodic_return_train',
            None, False, None, 'lower left'
        ]

    best_H = {
        'HalfCheetah-v2': 128,
        'Walker2d-v2': 256,
        'Hopper-v2': 1024,
        'Ant-v2': 64,
        'Humanoid-v2': 512,
        'HumanoidStandup-v2': 32,
    }

    best_gamma = {
        'HalfCheetah-v2': 0.99,
        'Walker2d-v2': 0.99,
        'Hopper-v2': 0.995,
        'Ant-v2': 0.97,
        'Humanoid-v2': 0.995,
        'HumanoidStandup-v2': 0.95,
    }

    Hs = [16, 32, 64, 128, 256, 512, 1024]
    data[6] = [
        [f'H_{H}-.*ppo_fhtd1' for H in Hs[:-1]] + ['active_1023.*ppo_fhtd2'],
        [f'PPO-FHTD H={H}' for H in Hs],
        'ppo-fhtd1',
        './log/discounting/neurips',
        'episodic_return_train',
        None, False, None, 'lower left'
    ]

    data[7] = [
        [f'active_{H - 1}-.*ppo_fhtd2' for H in Hs],
        [f'PPO-FHTD H={H}' for H in Hs],
        'ppo-fhtd2',
        './log/discounting/neurips',
        'episodic_return_train',
        None, False, None, 'lower left'
    ]

    # plot_paper_impl(*data[0])
    p = Pool(10)
    p.starmap(plot_paper_impl, [data[i] for i in range(8)])


def plot_paper_discounted(game, figure_size, params={}):
    plotter = Plotter()

    # game = 'Ant-v2'
    # game = 'HalfCheetah-v2'

    def get_pattern(discount):
        return [
            ['d_scheme_no-discount_%s-.*discounted_return_train' % (discount), 'flip_r'],
            ['aux_False-d_scheme_unbias-discount_%s-' % (discount), 'flip_r'],
            ['aux_True-d_scheme_unbias-discount_%s-.*sync_aux_True' % (discount), 'flip_r'],
            ['d_scheme_no-discount_%s-flip_r.*discounted_return_train' % (discount), ' '],
            ['aux_False-d_scheme_unbias-discount_%s-flip_r' % (discount), ' '],
            ['aux_True-d_scheme_unbias-discount_%s-flip_r.*sync_aux_True' % (discount), ' '],
        ]

    labels = [
        'PPO',
        'DisPPO',
        'AuxPPO',
    ]

    fontsize = 20

    def plot_games(self, gammas, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(gammas)
        if figure_size == 'small':
            plt.figure(figsize=(l * 5, 5))
        elif figure_size == 'large':
            plt.figure(figsize=(3 * 5, 2 * 5))
        else:
            raise NotImplementedError
        for i, gamma in enumerate(gammas):
            if figure_size == 'small':
                plt.subplot(1, l, i + 1)
            elif figure_size == 'large':
                plt.subplot(2, 3, i + 1)
            else:
                raise NotImplementedError
            patterns = get_pattern(gamma)
            for j, p in enumerate(patterns):
                label = labels[j] if j < 3 else None
                color = self.COLORS[j % 3]
                linestyle = 'solid' if j < 3 else (0, (5, 10))
                marker = None if j < 3 else 'D'
                positve_p, negative_p = (p, ' ') if isinstance(p, str) else p
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, positve_p),
                                                negative_pattern='.*%s.*' % (negative_p),
                                                **kwargs)
                kwargs['tag'] = 'discounted_return_train'
                x, y = self.load_results(log_dirs, **kwargs)
                assert y.shape[0] == 30, y.shape[0]
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                assert y.shape[0] == 30, y.shape[0]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se', markevery=10, marker=marker)
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            if i == 0:
                plt.ylabel(kwargs['ylabel'], fontsize=fontsize)
            plt.xlabel('steps', fontsize=fontsize)
            plt.xticks([0, 2e6], [0, '2M'], fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)
            plt.yscale(kwargs['yscale'])
            plt.title(r'%s $\gamma = %s$' % (game[:-3], gamma), fontsize=fontsize)
            if i == 0:
                plt.legend(fontsize=16)

    params.setdefault('downsample', 100)
    params.setdefault('window', 10)
    params.setdefault('interpolation', 100)
    params.setdefault('ylabel', 'discounted return')
    params.setdefault('yscale', 'linear')
    if 'Ant' in game:
        root = '/Volumes/Data/DeepRL/log/discounting/timestamp'
    elif 'HalfCheetah' in game:
        root = '/Volumes/Data/DeepRL/log/discounting/iclr_rebuttal/ppo42'
    plot_games(plotter,
               gammas=[0.995, 0.99, 0.97, 0.95, 0.93, 0.9],
               agg='mean',
               labels=labels,
               right_align=False,
               root=root,
               **params,
               )

    # plt.show()
    plt.tight_layout()
    if figure_size == 'small':
        filename = '/Users/Shangtong/GoogleDrive/Paper/discounting/img/aux_ppo_%s.pdf' % (game)
    elif figure_size == 'large':
        filename = '/Users/Shangtong/GoogleDrive/Paper/discounting/img/aux_ppo_%s_large.pdf' % (game)
    else:
        raise NotImplementedError
    plt.savefig(filename, bbox_inches='tight')


def plot_paper_discounted_episode_length(game, figure_size, params={}):
    plotter = Plotter()

    # game = 'Ant-v2'
    # game = 'HalfCheetah-v2'

    def get_pattern(discount):
        return [
            ['d_scheme_no-discount_%s-.*discounted_return_train' % (discount), 'flip_r'],
            ['aux_False-d_scheme_unbias-discount_%s-' % (discount), 'flip_r'],
            ['aux_True-d_scheme_unbias-discount_%s-.*sync_aux_True' % (discount), 'flip_r'],
            ['d_scheme_no-discount_%s-flip_r.*discounted_return_train' % (discount), ' '],
            ['aux_False-d_scheme_unbias-discount_%s-flip_r' % (discount), ' '],
            ['aux_True-d_scheme_unbias-discount_%s-flip_r.*sync_aux_True' % (discount), ' '],
        ]

    labels = [
        'PPO',
        'DisPPO',
        'AuxPPO',
    ]

    fontsize = 20

    def plot_games(self, gammas, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(gammas)
        if figure_size == 'small':
            plt.figure(figsize=(l * 5, 5))
        elif figure_size == 'large':
            plt.figure(figsize=(3 * 5, 2 * 5))
        else:
            raise NotImplementedError
        for i, gamma in enumerate(gammas):
            if figure_size == 'small':
                plt.subplot(1, l, i + 1)
            elif figure_size == 'large':
                plt.subplot(2, 3, i + 1)
            else:
                raise NotImplementedError
            patterns = get_pattern(gamma)
            for j, p in enumerate(patterns):
                label = labels[j] if j < 3 else None
                color = self.COLORS[j % 3]
                linestyle = 'solid' if j < 3 else (0, (5, 10))
                marker = None if j < 3 else 'D'
                positve_p, negative_p = (p, ' ') if isinstance(p, str) else p
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, positve_p),
                                                negative_pattern='.*%s.*' % (negative_p),
                                                **kwargs)
                kwargs['tag'] = 'episode_len'
                x, y = self.load_results(log_dirs, **kwargs)
                assert y.shape[0] == 30
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                assert y.shape[0] == 30
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se', markevery=10, marker=marker)
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            if i == 0:
                plt.ylabel(kwargs['ylabel'], fontsize=fontsize)
            plt.xlabel('steps', fontsize=fontsize)
            plt.xticks([0, 2e6], [0, '2M'], fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)
            plt.yscale(kwargs['yscale'])
            plt.title(r'%s $\gamma = %s$' % (game[:-3], gamma), fontsize=fontsize)
            if i == 0:
                plt.legend(fontsize=16)

    params.setdefault('downsample', 100)
    params.setdefault('window', 10)
    params.setdefault('interpolation', 100)
    params.setdefault('ylabel', 'episode length')
    params.setdefault('yscale', 'linear')
    if game == 'Ant-v2':
        # root = './log/discounting/timestamp'
        root = '/Volumes/Data/DeepRL//log/discounting/timestamp'
    else:
        root = './log/discounting/rebuttal/ppo42'
    plot_games(plotter,
               gammas=[0.995, 0.99, 0.97, 0.95, 0.93, 0.9],
               agg='mean',
               labels=labels,
               right_align=False,
               root=root,
               **params,
               )

    # plt.show()
    plt.tight_layout()
    if figure_size == 'small':
        filename = '/Users/Shangtong/GoogleDrive/Paper/discounting/img/aux_ppo_%s.pdf' % (game)
    elif figure_size == 'large':
        filename = '/Users/Shangtong/GoogleDrive/Paper/discounting/img/aux_ppo_%s_episode_length_large.pdf' % (game)
    else:
        raise NotImplementedError
    plt.savefig(filename, bbox_inches='tight')


def lr_transfer(dir=None, filename=None):
    plotter = Plotter()
    root = './log/discounting/rebuttal/'
    root = os.path.join(root, dir)
    info = dict()
    tags = ['discounted_return_train', 'episodic_return_train']

    fields = ['lr_%d' % lr for lr in range(10)]

    def score_fn(ys):
        return np.mean([y[-100:] for y in ys])

    for tag in tags:
        info[tag] = plotter.reduce(root, tag, fields, score_fn)
    with open('deep_rl/extra_data/%s.pkl' % (filename), 'wb') as f:
        pickle.dump(info, f)


def lr_transfer_view(filename):
    with open('deep_rl/extra_data/%s.pkl' % (filename), 'rb') as f:
        info = pickle.load(f)
    info = info['episodic_return_train']
    for key in info.keys():
        print(key)
        print(list(reversed(np.argsort(info[key]['scores']))))


if __name__ == '__main__':
    mkdir('images')
    # plot_ppo()
    # plot_ddpg_td3()
    # plot_atari()

    # plot_impl()
    # lr_search('ppo47', 'ppo47', './log/discounting/lr_search/neurips')
    # lr_search('ppo45', 'ppo45', '/Volumes/Data/logs/discounting/neurips/lr_search')
    # lr_search('ppo43', 'ppo43', './log/discounting/icml_rebuttal')
    # lr_search('ppo41', 'ppo41', './log/discounting/rebuttal', 'extra_lr_search')
    # lr_search('ppo39', 'ppo39')
    # lr_search_view('ppo39')
    # lr_transfer('lr_search_group1', 'lr_search_group1')
    # lr_transfer_view('lr_search_group1')
    # lr_transfer('lr_search_group2', 'lr_search_group2')
    # lr_transfer_view('lr_search_group2')
    # compute_best_v_max(0.995)
    # plot_paper()
    # plot_paper_discounted('HalfCheetah-v2', 'small')
    # plot_paper_discounted('HalfCheetah-v2', 'large')
    # plot_paper_discounted('Ant-v2', 'small')
    # plot_paper_discounted('Ant-v2', 'large')
    plot_paper_discounted_episode_length('Ant-v2', 'large')
