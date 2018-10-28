from deep_rl import *

def foo(game, **kwargs):
    kwargs.setdefault('tag', foo.__name__)
    kwargs.setdefault('log_dir', get_default_log_dir(kwargs['tag']))
    config = Config()
    config.merge(kwargs)

def batch():
    cf = Config()
    cf.add_argument('--gpu', type=int, default=0)
    cf.add_argument('--run', type=int, default=3)
    cf.merge()

    algo_ind = cf.gpu
    run_ind = cf.run

def single_run(run, game, fn, tag, **kwargs):
    random_seed()
    log_dir = './log/dist_rl-%s/%s/%s-run-%d' % (game, fn.__name__, tag, run)
    fn(game=game, log_dir=log_dir, tag=tag, **kwargs)

def multi_runs(game, fn, tag, **kwargs):
    kwargs.setdefault('runs', 2)
    runs = kwargs['runs']
    if np.isscalar(runs):
        runs = np.arange(0, runs)
    kwargs.setdefault('parallel', False)
    if not kwargs['parallel']:
        for run in runs:
            single_run(run, game, fn, tag, **kwargs)
        return
    ps = [mp.Process(target=single_run, args=(run, game, fn, tag), kwargs=kwargs) for run in runs]
    for p in ps:
        p.start()
        time.sleep(1)
    for p in ps: p.join()


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()
    # select_device(0)
    # batch()
    select_device(-1)