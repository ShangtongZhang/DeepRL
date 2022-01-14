import random

from deep_rl import *
import argparse
import os
import re
from sparse_reward import *


def get_lr(i):
    lr_actors = 3e-4 * np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])
    lrs = []
    for lr_a in lr_actors:
        for r in [1, 3]:
            lrs.append([lr_a, lr_a * r])
    return lrs[i]


def set_lr(params):
    src = './deep_rl/data'
    optimal_lrs = dict(episodic_return_train={}, discounted_return_train={})
    for subdir, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.pkl'):
                with open(os.path.join(subdir, file), 'rb') as f:
                    info = pickle.load(f)
                for key in info.keys():
                    optimal_lrs[key].update(info[key])

    for algo, param in params:
        lr_type = param['lr_type']
        del param['lr_type']
        generate_tag(param)
        tag = re.sub(r'lr_\d+', 'placeholder', param['tag'])
        tag = (re.sub(r'run.*', 'run', tag))
        tag = re.sub(r'.*-v2', 'Ant-v2', tag)
        tag = 'logger-%s' % (tag)
        del param['tag']
        field = optimal_lrs[lr_type][tag]['field']
        lr = int(field[3:])
        param['lr'] = lr
        param['lr_type'] = lr_type


def set_game_specific_lr(params, src=None):
    if src is None:
        src = './deep_rl/data/game_specific_lr'
    optimal_lrs = dict(episodic_return_train={}, discounted_return_train={})
    for subdir, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.pkl'):
                with open(os.path.join(subdir, file), 'rb') as f:
                    info = pickle.load(f)
                for key in info.keys():
                    optimal_lrs[key].update(info[key])

    for algo, param in params:
        lr_type = param['lr_type']
        del param['lr_type']
        generate_tag(param)
        tag = re.sub(r'lr_\d+', 'placeholder', param['tag'])
        tag = (re.sub(r'run.*', 'run', tag))
        # tag = re.sub(r'.*-v2', 'Ant-v2', tag)
        tag = 'logger-%s' % (tag)
        del param['tag']
        field = optimal_lrs[lr_type][tag]['field']
        lr = int(field[3:])
        param['lr'] = lr
        param['lr_type'] = lr_type


def batch_ppo3(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        # 'Swimmer-v2',
        'Hopper-v2',
        # 'Reacher-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for game in games:
        # for r in range(4, 6):
        for r in range(2, 4):
            # for r in range(0, 2):
            for d_scheme in [Config.NO, Config.UNBIAS, Config.COMP_UNBIAS, Config.INV_LINEAR, Config.LOG_LINEAR]:
                params.append([discount_ppo, dict(
                    game=game, run=r, remark='dppo', discount=0.99,
                    d_scheme=d_scheme, aux=False
                )])
            params.append([discount_ppo, dict(
                game=game, run=r, remark='dppo', discount=0.99,
                d_scheme=Config.UNBIAS, aux=True, sync_aux=False
            )])
            params.append([discount_ppo, dict(
                game=game, run=r, remark='dppo', discount=0.99,
                d_scheme=Config.UNBIAS, aux=True, sync_aux=True
            )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo4(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        # 'Swimmer-v2',
        'Hopper-v2',
        # 'Reacher-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 2):
    for r in range(6, 8):
        for game in games:
            for discount in [0.95, 0.97, 0.995]:
                for d_scheme in [Config.NO, Config.UNBIAS]:
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=d_scheme, aux=False
                    )])
                params.append([discount_ppo, dict(
                    game=game, run=r, remark='dppo', discount=discount,
                    d_scheme=Config.UNBIAS, aux=True, sync_aux=False
                )])
                params.append([discount_ppo, dict(
                    game=game, run=r, remark='dppo', discount=discount,
                    d_scheme=Config.UNBIAS, aux=True, sync_aux=True
                )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo5(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        # 'Ant-v2',
        'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 6):
        # for r in range(4, 6):
        # for r in range(2, 4):
        # for r in range(0, 2):
        for game in games:
            for lr in range(14):
                for discount in [0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=False, gae_tau=0, lr=lr,
                    )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                    )])
                for d_scheme in [Config.COMP_UNBIAS, Config.INV_LINEAR, Config.LOG_LINEAR]:
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=0.99,
                        d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                    )])
                params.append([discount_ppo, dict(
                    game=game, run=r, remark='dppo', discount=1,
                    d_scheme=Config.NO, aux=False, gae_tau=0, lr=lr,
                )])
                for H in [16, 32, 64, 128, 256]:
                    params.append(
                        [ppo_fhtd, dict(game=game, run=r, remark='fhppo', discount=1, H=H, gae_tau=0, lr=lr)])
                for v_max in [20, 40, 80, 160, 320, 640]:
                    params.append(
                        [ppo_c51, dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo6(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 3):
    # for r in range(3, 6):
    # for r in range(6, 8):
    for r in range(8, 10):
        for game in games:
            for lr in range(1):
                for discount in [0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=False, gae_tau=0, lr=lr,
                    )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                    )])
                for d_scheme in [Config.COMP_UNBIAS, Config.INV_LINEAR, Config.LOG_LINEAR]:
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=0.99,
                        d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                    )])
                params.append([discount_ppo, dict(
                    game=game, run=r, remark='dppo', discount=1,
                    d_scheme=Config.NO, aux=False, gae_tau=0, lr=lr,
                )])
                for H in [16, 32, 64, 128, 256]:
                    params.append(
                        [ppo_fhtd, dict(game=game, run=r, remark='fhppo', discount=1, H=H, gae_tau=0, lr=lr)])
                for v_max in [20, 40, 80, 160, 320, 640]:
                    params.append(
                        [ppo_c51, dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr)])

    with open('deep_rl/data/lr_search.pkl', 'rb') as f:
        info = pickle.load(f)['episodic_return_train']

    for algo, param in params:
        generate_tag(param)
        tag = re.sub(r'lr_\d+', 'placeholder', param['tag'])
        tag = (re.sub(r'run.*', 'run', tag))
        tag = re.sub(r'.*-v2', 'Ant-v2', tag)
        tag = 'logger-%s' % (tag)
        del param['tag']
        field = info[tag]['field']
        lr = int(field[3:])
        param['lr'] = lr

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo7(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(2, 4):
        # for r in range(0, 2):
        # for r in range(2, 4):
        # for r in range(0, 2):
        for game in games:
            for lr in range(14):
                for discount in [0.5, 0.7, 0.9, 0.93]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=False, gae_tau=0, lr=lr,
                    )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                    )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo8(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 3):
    # for r in range(3, 6):
    # for r in range(6, 9):
    for r in range(9, 10):
        for game in games:
            for lr in range(1):
                for discount in [0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=False, gae_tau=0, lr=lr,
                    )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                    )])

    with open('deep_rl/data/lr_search.pkl', 'rb') as f:
        info = pickle.load(f)['discounted_return_train']

    for algo, param in params:
        generate_tag(param)
        tag = re.sub(r'lr_\d+', 'placeholder', param['tag'])
        tag = (re.sub(r'run.*', 'run', tag))
        tag = re.sub(r'.*-v2', 'Ant-v2', tag)
        tag = 'logger-%s' % (tag)
        del param['tag']
        field = info[tag]['field']
        lr = int(field[3:])
        param['lr'] = lr

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo9(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(3, 4):
        # for r in range(2, 3):
        # for r in range(1, 2):
        # for r in range(0, 1):
        for game in games:
            for lr in range(10):
                for discount in [0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=False, gae_tau=0, lr=lr,
                    )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                    )])
                # for d_scheme in [Config.COMP_UNBIAS, Config.INV_LINEAR, Config.LOG_LINEAR]:
                #     params.append([discount_ppo, dict(
                #         game=game, run=r, remark='dppo', discount=0.99,
                #         d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                #     )])
                params.append([discount_ppo, dict(
                    game=game, run=r, remark='dppo', discount=1,
                    d_scheme=Config.NO, aux=False, gae_tau=0, lr=lr,
                )])
                for H in [16, 32, 64, 128, 256, 512, 1024]:
                    params.append(
                        [ppo_fhtd, dict(game=game, run=r, remark='fhppo', discount=1, H=H, gae_tau=0, lr=lr)])
                for v_max in [20, 40, 80, 160, 320, 640, 1280, 2560]:
                    params.append(
                        [ppo_c51, dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo10(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(9, 10):
        # for r in range(6, 9):
        # for r in range(3, 6):
        # for r in range(0, 3):
        for game in games:
            for lr in range(1):
                for discount in [0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, lr_type='discounted_return_train',
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=False, gae_tau=0, lr=lr,
                        lr_type='discounted_return_train',
                    )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                        lr_type='discounted_return_train',
                    )])
                for discount in [0.95, 0.97, 0.99, 0.995, 1]:
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.NO, aux=False, gae_tau=0, lr=lr, lr_type='episodic_return_train',
                    )])
                for H in [16, 32, 64, 128, 256, 512, 1024]:
                    params.append(
                        [ppo_fhtd, dict(game=game, run=r, remark='fhppo', discount=1, H=H, gae_tau=0, lr=lr,
                                        lr_type='episodic_return_train')])
                for v_max in [20, 40, 80, 160, 320, 640, 1280, 2560]:
                    params.append(
                        [ppo_c51, dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr,
                                       lr_type='episodic_return_train')])

    with open('deep_rl/data/lr_search.pkl', 'rb') as f:
        info = pickle.load(f)

    for algo, param in params:
        lr_type = param['lr_type']
        del param['lr_type']
        generate_tag(param)
        tag = re.sub(r'lr_\d+', 'placeholder', param['tag'])
        tag = (re.sub(r'run.*', 'run', tag))
        tag = re.sub(r'.*-v2', 'Ant-v2', tag)
        tag = 'logger-%s' % (tag)
        del param['tag']
        field = info[lr_type][tag]['field']
        lr = int(field[3:])
        param['lr'] = lr
        param['lr_type'] = lr_type

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo11(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(2, 3):
    for r in range(0, 2):
        for game in games:
            for lr in range(10):
                for H in [16, 32, 64, 128, 256, 512, 1024]:
                    params.append(
                        [ppo_fhtd, dict(game=game, run=r, remark='fhppo', discount=1, H=H, gae_tau=0, lr=lr, aux=True)])
                for v_max in [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 81920]:
                    params.append(
                        [ppo_c51,
                         dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr, aux=True)])
                for v_max in [5120, 81920]:
                    params.append(
                        [ppo_c51, dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo12(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(8, 10):
        # for r in range(5, 8):
        # for r in range(0, 5):
        for game in games:
            for lr in range(1):
                for H in [16, 32, 64, 128, 256, 512, 1024]:
                    params.append(
                        [ppo_fhtd, dict(game=game, run=r, remark='fhppo', discount=1, H=H, gae_tau=0, lr=lr, aux=True,
                                        lr_type='episodic_return_train')])
                for v_max in [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 81920]:
                    params.append(
                        [ppo_c51,
                         dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr, aux=True,
                              lr_type='episodic_return_train')])
                for v_max in [5120, 81920]:
                    params.append(
                        [ppo_c51, dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr,
                                       lr_type='episodic_return_train')])

    with open('deep_rl/data/lr_search.pkl', 'rb') as f:
        info = pickle.load(f)

    for algo, param in params:
        lr_type = param['lr_type']
        del param['lr_type']
        generate_tag(param)
        tag = re.sub(r'lr_\d+', 'placeholder', param['tag'])
        tag = (re.sub(r'run.*', 'run', tag))
        tag = re.sub(r'.*-v2', 'Ant-v2', tag)
        tag = 'logger-%s' % (tag)
        del param['tag']
        field = info[lr_type][tag]['field']
        lr = int(field[3:])
        param['lr'] = lr
        param['lr_type'] = lr_type

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo13(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(2, 3):
        # for r in range(0, 2):
        for game in games:
            for lr in range(10):
                for discount in [0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                            adv_gamma=1,
                        )])
                for extra_data in [2, 4]:
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=1,
                        d_scheme=Config.NO, aux=False, gae_tau=0, lr=lr,
                        extra_data=extra_data,
                    )])
                for discount in [0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]:
                    flip_r = int(np.log(0.05) / np.log(discount))
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, flip_r=flip_r,
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                        flip_r=flip_r,
                    )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo14(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(4, 10):
    for r in range(0, 4):
        for game in games:
            for lr in range(1):
                for discount in [0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                            adv_gamma=1, lr_type='episodic_return_train',
                        )])
                for extra_data in [2, 4]:
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=1,
                        d_scheme=Config.NO, aux=False, gae_tau=0, lr=lr,
                        extra_data=extra_data, lr_type='episodic_return_train',
                    )])
                for discount in [0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995]:
                    flip_r = int(np.log(0.05) / np.log(discount))
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, flip_r=flip_r,
                            lr_type='discounted_return_train',
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                        flip_r=flip_r,
                        lr_type='discounted_return_train',
                    )])

    set_lr(params)

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo15(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(2, 3):
        # for r in range(1, 2):
        # for r in range(0, 1):
        for game in games:
            for lr in range(10):
                for extra_data in [2, 4]:
                    for H in [16, 32, 64, 128, 256, 512, 1024]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=1, H=H, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data)])
                    for v_max in [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 81920]:
                        params.append(
                            [ppo_c51,
                             dict(game=game, run=r, remark='cppo', discount=1, v_max=v_max, gae_tau=0, lr=lr,
                                  aux=False, extra_data=extra_data)])
                for data_scale in [3, 5]:
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=1,
                        d_scheme=Config.NO, aux=False, gae_tau=0, lr=lr,
                        data_scale=data_scale
                    )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo16(cf):
    pass


def batch_ppo17(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(2, 3):
        # for r in range(1, 2):
        # for r in range(0, 1):
        for game in games:
            for lr in range(10):
                for extra_data in [0]:
                    for discount in [0.95, 0.97, 0.99, 0.995, 1]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                                critic_update='td', extra_data=extra_data
                            )])

    # params = params[25:]
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo18(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(5, 10):
        # for r in range(0, 5):
        for game in games:
            for lr in range(1):
                for extra_data in [0]:
                    for discount in [0.95, 0.97, 0.99, 0.995, 1]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                                critic_update='td', extra_data=extra_data, lr_type='episodic_return_train',
                            )])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo19(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(2, 3):
        # for r in range(0, 2):
        for game in games:
            for lr in range(10):
                for extra_data in [2, 4]:
                    for discount in [1]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                                critic_update='td', extra_data=extra_data
                            )])

    # params = params[25:]
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo20(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(8, 10):
        # for r in range(4, 8):
        # for r in range(0, 4):
        for game in games:
            for lr in range(1):
                for extra_data in [2, 4]:
                    for discount in [1]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                                critic_update='td', extra_data=extra_data, lr_type='episodic_return_train'
                            )])
    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo21(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 3):
        for game in games:
            # for lr in range(5, 10):
            for lr in range(0, 5):
                for extra_data in [0]:
                    for H in [16, 32, 64, 128, 256, 512, 1024]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=0.99, H=H, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data)])
                    for v_max in [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 81920]:
                        params.append(
                            [ppo_c51,
                             dict(game=game, run=r, remark='cppo', discount=0.99, v_max=v_max, gae_tau=0, lr=lr,
                                  aux=False, extra_data=extra_data)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo22(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 5):
    for r in range(5, 10):
        for game in games:
            for lr in range(1):
                for extra_data in [0]:
                    for H in [16, 32, 64, 128, 256, 512, 1024]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=0.99, H=H, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data, lr_type='episodic_return_train')])
                    for v_max in [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 81920]:
                        params.append(
                            [ppo_c51,
                             dict(game=game, run=r, remark='cppo', discount=0.99, v_max=v_max, gae_tau=0, lr=lr,
                                  aux=False, extra_data=extra_data, lr_type='episodic_return_train')])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo23(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(2, 3):
        # for r in range(1, 2):
        # for r in range(0, 1):
        for game in games:
            for lr in range(0, 10):
                for extra_data in [0]:
                    for active in [64, 128, 256, 512, 1024]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=0.99, H=1024, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data, active=active - 1)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo24(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(6, 10):
    # for r in range(3, 6):
    for r in range(0, 3):
        # for r in range(0, 10):
        for game in games:
            for lr in range(1):
                for extra_data in [0]:
                    for active in [64, 128, 256, 512, 1024]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=0.99, H=1024, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data, active=active - 1, lr_type='episodic_return_train')])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo25(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 3):
        for game in games:
            for lr in range(7, 10):
                for extra_data in [0]:
                    for active in [64, 128, 256, 512, 1024]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=1, H=1024, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data, active=active - 1)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo26(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(9, 10):
        # for r in range(6, 9):
        # for r in range(0, 6):
        for game in games:
            for lr in range(1):
                for extra_data in [0]:
                    for active in [64, 128, 256, 512, 1024]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=1, H=1024, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data, active=active - 1, lr_type='episodic_return_train')])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo27(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(2, 3):
        # for r in range(0, 2):
        for game in games:
            for lr in range(0, 10):
                for extra_data in [0]:
                    for v_max in [10240, 81920, 163840, 327680]:
                        for discount in [0.99, 1]:
                            params.append(
                                [ppo_c51,
                                 dict(game=game, run=r, remark='cppo', discount=discount, v_max=v_max, gae_tau=0, lr=lr,
                                      aux=False, extra_data=extra_data)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo28(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(7, 10):
        for game in games:
            for lr in range(1):
                for extra_data in [0]:
                    for v_max in [10240, 81920, 163840, 327680]:
                        for discount in [0.99, 1]:
                            params.append(
                                [ppo_c51,
                                 dict(game=game, run=r, remark='cppo', discount=discount, v_max=v_max, gae_tau=0, lr=lr,
                                      aux=False, extra_data=extra_data, lr_type='episodic_return_train')])
    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo29(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 3):
        for game in games:
            for lr in range(10):
                for discount in [0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                            adv_gamma=1, critic_update='td',
                        )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo30(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 10):
        for game in games:
            for lr in range(1):
                for discount in [0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                            adv_gamma=1, critic_update='td', lr_type='episodic_return_train'
                        )])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo31(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    # with open('./deep_rl/data_plot/best_v_max.pkl', 'rb') as f:
    #     best_v_max = pickle.load(f)

    params = []

    for r in range(2, 3):
        for game in games:
            for lr in range(0, 10):
                for extra_data in [0]:
                    for v_max in [320, 640, 1280, 2560, 5120, 10240, 81920, 163840, 327680]:
                        for discount in [0.995]:
                            params.append(
                                [ppo_c51,
                                 dict(game=game, run=r, remark='cppo', discount=discount, v_max=v_max, gae_tau=0, lr=lr,
                                      aux=False, extra_data=extra_data)])
                for extra_data in [2, 4]:
                    for discount in [0.99, 0.995]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                                critic_update='td', extra_data=extra_data
                            )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo32(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 10):
        for game in games:
            for lr in range(1):
                for extra_data in [0]:
                    for v_max in [320, 640, 1280, 2560, 5120, 10240, 81920, 163840, 327680]:
                        for discount in [0.995]:
                            params.append(
                                [ppo_c51,
                                 dict(game=game, run=r, remark='cppo', discount=discount, v_max=v_max, gae_tau=0, lr=lr,
                                      aux=False, extra_data=extra_data, lr_type='episodic_return_train')])
                for extra_data in [2, 4]:
                    for discount in [0.99, 0.995]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                                critic_update='td', extra_data=extra_data, lr_type='episodic_return_train'
                            )])
    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo33(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(2, 3):
        # for r in range(0, 2):
        for game in games:
            for lr in range(10):
                for discount in [0.99, 0.995, 1]:
                    for mp in [1, 2, 4]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                                critic_update='td', multi_path=mp, adv_gamma=1,
                            )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo34(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 10):
        for game in games:
            for lr in range(1):
                # for discount in [0.99]:
                # for discount in [0.995, 1]:
                for discount in [0.99, 0.995, 1]:
                    for mp in [1, 2, 4]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, adv_gamma=1,
                                critic_update='td', multi_path=mp, lr_type='episodic_return_train'
                            )])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo35(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 3):
        for game in games:
            for lr in range(10):
                for discount in [0.99, 0.995, 1]:
                    for nq in [25, 50, 100]:
                        params.append([ppo_qr, dict(
                            game=game, run=r, remark='qrppo', discount=discount,
                            aux=False, gae_tau=0, lr=lr,
                            critic_update='td', num_quantiles=nq
                        )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo36(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 10):
        for game in games:
            for lr in range(1):
                for discount in [0.99, 0.995, 1]:
                    for nq in [25, 50, 100]:
                        params.append([ppo_qr, dict(
                            game=game, run=r, remark='qrppo', discount=discount,
                            aux=False, gae_tau=0, lr=lr,
                            critic_update='td', num_quantiles=nq, lr_type='episodic_return_train'
                        )])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo37(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 3):
        for game in games:
            for lr in range(0, 10):
                for extra_data in [0]:
                    for active in [16, 32]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=1, H=1024, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data, active=active - 1)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo38(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 5):
    for r in range(5, 10):
        for game in games:
            for lr in range(1):
                for extra_data in [0]:
                    for active in [16, 32]:
                        params.append(
                            [ppo_fhtd,
                             dict(game=game, run=r, remark='fhppo', discount=1, H=1024, gae_tau=0, lr=lr, aux=False,
                                  extra_data=extra_data, active=active - 1, lr_type='episodic_return_train')])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo39(cf):
    "lr search on more games in addition to ppo17 expts"
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        # 'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 3):
        # for r in range(1, 2):
        # for r in range(0, 1):
        for game in games:
            for lr in range(10):
                for extra_data in [0]:
                    # for discount in [0.95, 0.97, 0.99, 0.995, 1]:
                    for discount in [0.995]:
                        # for discount in [0.99]:
                        for d_scheme in [Config.NO]:
                            params.append([discount_ppo, dict(
                                game=game, run=r, remark='dppo', discount=discount,
                                d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                                critic_update='td', extra_data=extra_data
                            )])

    # params = params[25:]
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo41(cf):
    games = [
        'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        # 'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 3):
    for r in range(2, 3):
        # for r in range(1, 2):
        # for r in range(0, 1):
        for game in games:
            for lr in range(10):
                for discount in [0.9, 0.93, 0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr,
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                    )])
                    flip_r = int(np.log(0.05) / np.log(discount))
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, flip_r=flip_r,
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                        flip_r=flip_r,
                    )])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo42(cf):
    games = [
        'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        # 'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 5):
    # for r in range(5, 10):
    for r in range(10, 30):
        for game in games:
            for lr in range(1):
                for discount in [0.9, 0.93, 0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, lr_type='discounted_return_train'
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                        lr_type='discounted_return_train'
                    )])
                    flip_r = int(np.log(0.05) / np.log(discount))
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, flip_r=flip_r,
                            lr_type='discounted_return_train'
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                        flip_r=flip_r, lr_type='discounted_return_train'
                    )])

    set_game_specific_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo42_prime(cf):
    games = [
        # 'HalfCheetah-v2',
        # 'Walker2d-v2',
        # 'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 5):
    # for r in range(5, 10):
    for r in range(10, 30):
        for game in games:
            for lr in range(1):
                for discount in [0.9, 0.93, 0.95, 0.97, 0.99, 0.995]:
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, lr_type='discounted_return_train'
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                        lr_type='discounted_return_train'
                    )])
                    flip_r = int(np.log(0.05) / np.log(discount))
                    for d_scheme in [Config.NO, Config.UNBIAS]:
                        params.append([discount_ppo, dict(
                            game=game, run=r, remark='dppo', discount=discount,
                            d_scheme=d_scheme, aux=False, gae_tau=0, lr=lr, flip_r=flip_r,
                            lr_type='discounted_return_train'
                        )])
                    params.append([discount_ppo, dict(
                        game=game, run=r, remark='dppo', discount=discount,
                        d_scheme=Config.UNBIAS, aux=True, sync_aux=True, gae_tau=0, lr=lr,
                        flip_r=flip_r, lr_type='discounted_return_train'
                    )])

    set_lr(params)
    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo43(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 3):
        for game in games:
            for lr in range(0, 10):
                # for lr in range(5, 10):
                for discount in [0.99, 1]:
                    for use_target_net in [True, False]:
                        params.append(
                            [discount_ppo,
                             dict(game=game, run=r, remark='dppo', discount=discount, gae_tau=0, lr=lr,
                                  d_scheme=Config.NO,
                                  critic_update='td', use_target_net=use_target_net)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo44(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        # 'Humanoid-v2',
        # 'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 10):
        for game in games:
            for lr in range(0, 1):
                # for discount in [0.99, 1]:
                for discount in [1]:
                    for use_target_net in [True, False]:
                        params.append(
                            [discount_ppo,
                             dict(game=game, run=r, remark='dppo', discount=discount, gae_tau=0, lr=lr,
                                  d_scheme=Config.NO,
                                  critic_update='td', use_target_net=use_target_net,
                                  lr_type='episodic_return_train')])

    if cf.i >= len(params):
        exit()

    set_lr(params)
    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo45(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    for r in range(0, 3):
        for game in games:
            # for lr in range(0, 5):
            for lr in range(5, 10):
                for discount in [0.95, 0.97, 0.99, 0.995, 1]:
                    params.append(
                        [discount_ppo,
                         dict(game=game, run=r, remark='ppo', discount=discount, gae_tau=0, lr=lr, d_scheme=Config.NO,
                              critic_update='mc', multi_path=0)])
                    params.append(
                        [discount_ppo,
                         dict(game=game, run=r, remark='ppo_td', discount=discount, gae_tau=0, lr=lr,
                              d_scheme=Config.NO,
                              critic_update='td', multi_path=0)])
                    params.append(
                        [discount_ppo,
                         dict(game=game, run=r, remark='dis_ppo', discount=discount, gae_tau=0, lr=lr,
                              d_scheme=Config.UNBIAS,
                              critic_update='td', multi_path=0)])
                for discount in [0.99, 0.995, 1]:
                    for mp in [0, 1, 2, 4]:
                        params.append(
                            [discount_ppo,
                             dict(game=game, run=r, remark='ppo_td_ex', discount=discount, gae_tau=0, lr=lr,
                                  d_scheme=Config.NO,
                                  critic_update='td', multi_path=mp, adv_gamma=1)])
                for H in [16, 32, 64, 128, 256, 512, 1024]:
                    params.append(
                        [ppo_fhtd,
                         dict(game=game, run=r, remark='ppo_fhtd1', discount=1, H=H, gae_tau=0, lr=lr, active=-1)])
                    params.append(
                        [ppo_fhtd,
                         dict(game=game, run=r, remark='ppo_fhtd2', discount=1, H=1024, gae_tau=0, lr=lr,
                              active=H - 1)])

    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo46(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 5):
    # for r in range(5, 8):
    # for r in range(8, 10):
    # for r in range(10, 20):
    for r in range(20, 30):
        for game in games:
            for lr in range(0, 1):
                for discount in [0.95, 0.97, 0.99, 0.995, 1]:
                    params.append(
                        [discount_ppo,
                         dict(game=game, run=r, remark='ppo', discount=discount, gae_tau=0, lr=lr, d_scheme=Config.NO,
                              critic_update='mc', multi_path=0, lr_type='episodic_return_train')])
                    params.append(
                        [discount_ppo,
                         dict(game=game, run=r, remark='ppo_td', discount=discount, gae_tau=0, lr=lr,
                              d_scheme=Config.NO,
                              critic_update='td', multi_path=0, lr_type='episodic_return_train')])
                    params.append(
                        [discount_ppo,
                         dict(game=game, run=r, remark='dis_ppo', discount=discount, gae_tau=0, lr=lr,
                              d_scheme=Config.UNBIAS,
                              critic_update='td', multi_path=0, lr_type='episodic_return_train')])
                for discount in [0.99, 0.995, 1]:
                    for mp in [0, 1, 2, 4]:
                        params.append(
                            [discount_ppo,
                             dict(game=game, run=r, remark='ppo_td_ex', discount=discount, gae_tau=0, lr=lr,
                                  d_scheme=Config.NO,
                                  critic_update='td', multi_path=mp, adv_gamma=1, lr_type='episodic_return_train')])
                for H in [16, 32, 64, 128, 256, 512, 1024]:
                    params.append(
                        [ppo_fhtd,
                         dict(game=game, run=r, remark='ppo_fhtd1', discount=1, H=H, gae_tau=0, lr=lr, active=-1, lr_type='episodic_return_train')])
                    params.append(
                        [ppo_fhtd,
                         dict(game=game, run=r, remark='ppo_fhtd2', discount=1, H=1024, gae_tau=0, lr=lr,
                              active=H - 1, lr_type='episodic_return_train')])

    if cf.i >= len(params):
        exit()

    set_game_specific_lr(params, src='./deep_rl/data/neurips')
    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo47(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 1):
    # for r in range(1, 2):
    for r in range(2, 3):
        for game in games:
            for lr in range(0, 10):
                for discount in [0.95, 0.97, 0.99, 0.995, 1]:
                    params.append(
                        [discount_ppo,
                         dict(game=game, run=r, remark='dis_ppo', discount=discount, gae_tau=0, lr=lr,
                              d_scheme=Config.UNBIAS,
                              critic_update='mc', multi_path=0)])


    if cf.i >= len(params):
        exit()

    algo, param = params[cf.i]
    algo(**param)

    exit()


def batch_ppo48(cf):
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    params = []

    # for r in range(0, 5):
    for r in range(5, 10):
        for game in games:
            for lr in range(0, 1):
                for discount in [0.95, 0.97, 0.99, 0.995, 1]:
                    # for ds in [Config.UNBIAS]:
                    #     params.append(
                    #         [discount_ppo,
                    #          dict(game=game, run=r, remark='dis_ppo', discount=discount, gae_tau=0, lr=lr,
                    #               d_scheme=ds,
                    #               critic_update='mc', multi_path=0, lr_type='discounted_return_train')])
                    for ds in [Config.NO]:
                        params.append(
                            [discount_ppo,
                             dict(game=game, run=r, remark='ppo', discount=discount, gae_tau=0, lr=lr,
                                  d_scheme=ds,
                                  critic_update='mc', multi_path=0, lr_type='discounted_return_train')])

    if cf.i >= len(params):
        exit()

    set_game_specific_lr(params, src='./deep_rl/data/neurips')
    algo, param = params[cf.i]
    algo(**param)

    exit()


def ppo_c51(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('discount', 0.99)
    kwargs.setdefault('categorical_n_atoms', 50)
    kwargs.setdefault('v_max', 200)
    kwargs.setdefault('critic_update', 'td')
    kwargs.setdefault('gae_tau', 0.95)
    kwargs.setdefault('use_gae', True)
    kwargs.setdefault('lr', 1)
    kwargs.setdefault('aux', False)
    kwargs.setdefault('extra_data', 0)
    config = Config()
    config.merge(kwargs)
    lr_actor, lr_critic = get_lr(config.lr)

    config.categorical_v_min = -config.v_max
    config.categorical_v_max = config.v_max

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: PPOC51Net(
        config.state_dim, config.action_dim,
        n_atoms=config.categorical_n_atoms,
        actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, lr_actor)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, lr_critic)
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 2e6
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOC51Agent(config))


def ppo_fhtd(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('discount', 0.99)
    kwargs.setdefault('extra_data', 0)
    kwargs.setdefault('H', 128)
    kwargs.setdefault('gae_tau', 0.95)
    kwargs.setdefault('use_gae', True)
    kwargs.setdefault('lr', 1)
    kwargs.setdefault('aux', False)
    kwargs.setdefault('active', -1)
    config = Config()
    config.merge(kwargs)
    lr_actor, lr_critic = get_lr(config.lr)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: PPOFHTDNet(
        config.state_dim, config.action_dim,
        H=config.H,
        actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(
        params, lr_actor)
    config.critic_opt_fn = lambda params: torch.optim.Adam(
        params, lr_critic)
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 2e6
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOFHTDAgent(config))


def ppo_qr(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('discount', 0.99)
    kwargs.setdefault('critic_update', 'td')
    kwargs.setdefault('gae_tau', 0.95)
    kwargs.setdefault('use_gae', True)
    kwargs.setdefault('lr', 1)
    kwargs.setdefault('aux', False)
    kwargs.setdefault('extra_data', 0)
    kwargs.setdefault('num_quantiles', 50)
    config = Config()
    config.merge(kwargs)
    lr_actor, lr_critic = get_lr(config.lr)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: PPOQRNet(
        config.state_dim, config.action_dim,
        n_quantiles=config.num_quantiles,
        actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, lr_actor)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, lr_critic)
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 2e6
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOQRAgent(config))


def discount_ppo(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('discount', 0.99)
    kwargs.setdefault('gae_tau', 0.95)
    kwargs.setdefault('use_gae', True)
    kwargs.setdefault('lr', 1)
    kwargs.setdefault('d_scheme', Config.NO)
    kwargs.setdefault('aux', False)
    kwargs.setdefault('sync_aux', False)
    kwargs.setdefault('flip_r', 0)
    kwargs.setdefault('extra_data', 0)
    kwargs.setdefault('adv_gamma', kwargs['discount'])
    kwargs.setdefault('data_scale', 1)
    kwargs.setdefault('critic_update', 'mc')
    kwargs.setdefault('multi_path', 0)
    kwargs.setdefault('use_target_net', False)
    config = Config()
    config.merge(kwargs)
    lr_actor, lr_critic = get_lr(config.lr)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(
            config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh), aux=True)
    config.actor_opt_fn = lambda params: torch.optim.Adam(
        params, lr_actor)
    config.critic_opt_fn = lambda params: torch.optim.Adam(
        params, lr_critic)
    config.gradient_clip = 0.5
    config.rollout_length = 2048 * config.data_scale
    config.optimization_epochs = 10 * config.data_scale
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048 * config.data_scale
    config.eval_episodes = 10
    config.max_steps = 2e6
    config.eval_interval = config.max_steps / 100
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    run_steps(DiscountPPOAgent(config))


if __name__ == '__main__':
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--logdir', type=str, default='.')
    cf.merge()

    set_base_log_dir(cf.logdir)

    mkdir(os.path.join(cf.logdir, 'log'))
    mkdir(os.path.join(cf.logdir, 'data'))
    random_seed()

    select_device(-1)
    # batch_ppo46(cf)
    # batch_ppo42(cf)
    batch_ppo42_prime(cf)

    # policy_gradient(
    #     game='MDP',
    #     # d_scheme=Config.UNBIAS,
    #     d_scheme=Config.NO,
    #     lr=2,
    # )

    game = 'HalfCheetah-v2'
    # game = 'Reacher-v2'
    # game = 'Hopper-v2'
    # ppo_continuous(
    #     game=game,
    #     discount=0.99,
    #     critic_update='mc',
    #     # gae_tau=0.1,
    #     use_gae=True,
    #     bootstrap_with_oracle=False,
    #     normalized_adv=True,
    #     use_oracle_v=False,
    #     mc_n=10,
    # )

    discount_ppo(
        game=game,
        d_scheme=Config.NO,
        aux=False,
        sync_aux=True,
        gae_tau=0,
        discount=0.99,
        lr=6,
        critic_update='td',
        multi_path=2,
        extra_data=0,
        adv_gamma=1,
        # data_scale=3,
        # adv_gamma=1,
        # extra_data=0,
        # discount=0.99,
        # lr=4,
    )

    # ppo_fhtd(
    #     game=game,
    #     discount=1,
    #     H=1024,
    #     active=64,
    #     gau_tau=0,
    #     aux=False,
    #     extra_data=2,
    # )

    # ppo_c51(
    #     game=game,
    #     discount=1,
    #     v_max=200,
    #     aux=False,
    #     extra_data=2,
    # )

    # ppo_qr(
    #     game=game,
    #     discount=1,
    #     lr=6,
    #     gae_tau=0,
    # )
