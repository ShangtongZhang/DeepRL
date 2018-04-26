import logging
from agent import *
from component import *
from utils import *
from model import *

def ddpg_continuous():
    config = Config()
    log_dir = get_default_log_dir(ddpg_continuous.__name__)
    # config.task_fn = lambda: Pendulum(log_dir=log_dir)
    # config.task_fn = lambda: Roboschool('RoboschoolInvertedPendulum-v1', log_dir=log_dir)
    # config.task_fn = lambda: Roboschool('RoboschoolReacher-v1', log_dir=log_dir)
    config.task_fn = lambda: Roboschool('RoboschoolHopper-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolAnt-v1', log_dir=log_dir)
    # config.task_fn = lambda: Roboschool('RoboschoolWalker2d-v1', log_dir=log_dir)
    # config.task_fn = lambda: DMControl('cartpole', 'balance', log_dir=log_dir)
    # config.task_fn = lambda: DMControl('finger', 'spin', log_dir=log_dir)
    config.evaluation_env = Roboschool('RoboschoolHopper-v1', log_dir=log_dir)
    config.actor_network_fn = lambda state_dim, action_dim: DeterministicActorNet(
        action_dim, TwoLayerFCBody(state_dim, [300, 200]))
    config.critic_network_fn = lambda state_dim, action_dim: DeterministicCriticNet(
        TwoLayerFCBodyWithAction(state_dim, action_dim, [400, 300]))
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.state_normalizer = RunningStatsNormalizer()
    config.max_steps = 1e6
    config.random_process_fn = \
        lambda action_dim: OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, sigma=0.3,
                                         n_steps_annealing=1000000)
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.gradient_clip = 1.0
    config.logger = Logger('./log', logger)
    run_episodes(DDPGAgent(config))

if __name__ == '__main__':
    mkdir('data')
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    os.system('export OMP_NUM_THREADS=1')
    os.system('export MKL_NUM_THREADS=1')
    torch.set_num_threads(1)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    ddpg_continuous()



