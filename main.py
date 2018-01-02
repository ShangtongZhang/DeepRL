#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import logging
from agent import *
from component import *
from utils import *
import model.action_conditional_video_prediction as acvp

def dqn_cart_pole():
    config = Config()
    config.task_fn = lambda: CartPole()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: FCNet([8, 50, 200, 2])
    # config.network_fn = lambda optimizer_fn: DuelingFCNet([8, 50, 200, 2], optimizer_fn)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.max_episode_length = 200
    config.exploration_steps = 1000
    config.logger = Logger('./log', logger)
    config.history_length = 2
    config.test_interval = 100
    config.test_repetitions = 50
    config.double_q = True
    # config.double_q = False
    run_episodes(DQNAgent(config))

def async_cart_pole():
    config = Config()
    config.task_fn= lambda: CartPole()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: FCNet([4, 50, 200, 2])
    config.policy_fn = lambda: GreedyPolicy(epsilon=0.5, final_step=5000, min_epsilon=0.1)
    # config.worker = OneStepQLearning
    config.worker = NStepQLearning
    # config.worker = OneStepSarsa
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.max_episode_length = 200
    config.num_workers = 16
    config.update_interval = 6
    config.test_interval = 1
    config.test_repetitions = 50
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def a3c_cart_pole():
    config = Config()
    config.task_fn = lambda: CartPole()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: ActorCriticFCNet(4, 2)
    config.policy_fn = SamplePolicy
    config.worker = AdvantageActorCritic
    config.discount = 0.99
    config.max_episode_length = 200
    config.num_workers = 16
    config.update_interval = 6
    config.test_interval = 1
    config.test_repetitions = 30
    config.logger = Logger('./log', logger)
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    agent = AsyncAgent(config)
    agent.run()

def dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, normalized_state=False)
    action_dim = config.task_fn().action_dim
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda: NatureConvNet(config.history_length, action_dim)
    # config.network_fn = lambda optimizer_fn: DuelingNatureConvNet(config.history_length, n_actions, optimizer_fn)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=32, dtype=np.uint8)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 0
    config.exploration_steps= 50000
    config.logger = Logger('./log', logger)
    config.test_interval = 10
    config.test_repetitions = 1
    # config.double_q = True
    config.double_q = False
    run_episodes(DQNAgent(config))

def async_pixel_atari(name):
    config = Config()
    config.history_length = 1
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, frame_size=42)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0001)
    config.network_fn = lambda: OpenAIConvNet(
        config.history_length, task.env.action_space.n)
    config.policy_fn = lambda: StochasticGreedyPolicy(
        epsilons=[0.7, 0.7, 0.7], final_step=2000000, min_epsilons=[0.1, 0.01, 0.5],
        probs=[0.4, 0.3, 0.3])
    # config.worker = OneStepSarsa
    # config.worker = NStepQLearning
    config.worker = OneStepQLearning
    config.reward_shift_fn = lambda r: np.sign(r)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 10000
    config.num_workers = 6
    config.update_interval = 20
    config.test_interval = 50000
    config.test_repetitions = 1
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def a3c_pixel_atari(name):
    config = Config()
    config.history_length = 1
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, frame_size=42)
    task = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0001)
    config.network_fn = lambda: OpenAIActorCriticConvNet(
        config.history_length, task.env.action_space.n, LSTM=True)
    config.reward_shift_fn = lambda r: np.sign(r)
    config.policy_fn = SamplePolicy
    config.worker = AdvantageActorCritic
    config.discount = 0.99
    config.max_episode_length = 10000
    config.num_workers = 6
    config.update_interval = 20
    config.test_interval = 50000
    config.test_repetitions = 1
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def dqn_fruit():
    config = Config()
    config.task_fn = lambda: Fruit()
    config.optimizer_fn = lambda params: torch.optim.SGD(params, 0.01, momentum=0.9)
    config.reward_weight = np.ones(10) / 10
    config.hybrid_reward = False
    config.network_fn = lambda: FruitHRFCNet(98, 4, config.reward_weight)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=15)
    config.discount = 0.95
    config.target_network_update_freq = 200
    config.max_episode_length = 100
    config.exploration_steps = 200
    config.logger = Logger('./log', logger)
    config.history_length = 1
    config.test_interval = 0
    config.test_repetitions = 10
    config.episode_limit = 5000
    config.double_q = False
    run_episodes(DQNAgent(config))

def hrdqn_fruit():
    config = Config()
    config.task_fn = lambda: Fruit(hybrid_reward=True)
    config.hybrid_reward = True
    config.reward_weight = np.ones(10) / 10
    config.optimizer_fn = lambda params: torch.optim.SGD(params, 0.01, momentum=0.9)
    config.network_fn = lambda optimizer_fn: FruitHRFCNet(98, 4, config.reward_weight)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=10000, min_epsilon=0.1)
    config.replay_fn = lambda: HybridRewardReplay(memory_size=10000, batch_size=15)
    config.discount = 0.95
    config.target_network_update_freq = 200
    config.max_episode_length = 100
    config.exploration_steps = 200
    config.logger = Logger('./log', logger)
    config.history_length = 1
    config.test_interval = 0
    config.test_repetitions = 10
    config.target_type = config.expected_sarsa_target
    # config.target_type = config.q_target
    config.double_q = False
    config.episode_limit = 5000
    run_episodes(DQNAgent(config))

def a3c_continuous():
    config = Config()
    config.task_fn = lambda: Pendulum()
    # config.task_fn = lambda: BipedalWalkerHardcore()
    task = config.task_fn()
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, 0.0001)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda: DisjointActorCriticNet(
        # lambda: GaussianActorNet(task.state_dim, task.action_dim, unit_std=False, action_gate=F.tanh, action_scale=2.0),
        lambda: GaussianActorNet(task.state_dim, task.action_dim, unit_std=True),
        lambda: GaussianCriticNet(task.state_dim))
    config.policy_fn = lambda: GaussianPolicy()
    config.worker = ContinuousAdvantageActorCritic
    config.discount = 0.99
    config.max_episode_length = task.max_episode_steps
    config.num_workers = 8
    config.update_interval = 20
    config.test_interval = 1
    config.test_repetitions = 1
    config.entropy_weight = 0
    config.gradient_clip = 40
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def p3o_continuous():
    config = Config()
    config.task_fn = lambda: Pendulum()
    # config.task_fn = lambda: BipedalWalker()
    # config.task_fn = lambda: BipedalWalkerHardcore()
    # config.task_fn = lambda: Roboschool('RoboschoolInvertedPendulum-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolAnt-v1')
    task = config.task_fn()
    config.actor_network_fn = lambda: GaussianActorNet(task.state_dim, task.action_dim,
                                                       gpu=False, unit_std=True)
    config.critic_network_fn = lambda: GaussianCriticNet(task.state_dim, gpu=False)
    config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)

    config.policy_fn = lambda: GaussianPolicy()
    config.replay_fn = lambda: GeneralReplay(memory_size=2048, batch_size=2048)
    config.worker = ProximalPolicyOptimization
    config.discount = 0.99
    config.gae_tau = 0.97
    config.num_workers = 6
    config.test_interval = 1
    config.test_repetitions = 1
    config.max_episode_length = task.max_episode_steps
    config.entropy_weight = 0
    config.gradient_clip = 20
    config.rollout_length = 10000
    config.optimize_epochs = 1
    config.ppo_ratio_clip = 0.2
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def d3pg_continuous():
    config = Config()
    config.task_fn = lambda: Pendulum()
    # config.task_fn = lambda: ContinuousLunarLander()
    # config.task_fn = lambda: Roboschool('RoboschoolInvertedPendulum-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolReacher-v1')
    # config.task_fn = lambda: BipedalWalker()
    task = config.task_fn()
    config.actor_network_fn = lambda: DeterministicActorNet(
        task.state_dim, task.action_dim, F.tanh, 2, non_linear=F.relu, batch_norm=False)
    config.critic_network_fn = lambda: DeterministicCriticNet(
        task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=False)
    config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.critic_optimizer_fn =\
        lambda params: torch.optim.Adam(params, lr=1e-4)
    config.replay_fn = lambda: SharedReplay(memory_size=1000000, batch_size=64,
                                            state_shape=(task.state_dim, ), action_shape=(task.action_dim, ))
    config.discount = 0.99
    config.max_episode_length = task.max_episode_steps
    config.random_process_fn = \
        lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.15, sigma=0.2,
                                         n_steps_annealing=100000)
    config.worker = DeterministicPolicyGradient
    config.num_workers = 6
    config.min_memory_size = 50
    config.target_network_mix = 0.001
    config.test_interval = 500
    config.test_repetitions = 1
    config.gradient_clip = 20
    config.logger = Logger('./log', logger)
    agent = AsyncAgent(config)
    agent.run()

def ddpg_continuous():
    config = Config()
    # config.task_fn = lambda: Pendulum()
    # config.task_fn = lambda: ContinuousLunarLander()
    # config.task_fn = lambda: Roboschool('RoboschoolInvertedPendulum-v1')
    config.task_fn = lambda: Roboschool('RoboschoolReacher-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolHopper-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolAnt-v1')
    # config.task_fn = lambda: Roboschool('RoboschoolWalker2d-v1')
    # config.task_fn = lambda: BipedalWalker()
    task = config.task_fn()
    config.actor_network_fn = lambda: DeterministicActorNet(
        task.state_dim, task.action_dim, F.tanh, 1, non_linear=F.relu, batch_norm=False)
    config.critic_network_fn = lambda: DeterministicCriticNet(
        task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=False)
    config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
    config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    config.critic_optimizer_fn =\
        lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
    config.replay_fn = lambda: SharedReplay(memory_size=1000000, batch_size=64,
                                            state_shape=(task.state_dim, ), action_shape=(task.action_dim, ))
    config.discount = 0.99
    config.max_episode_length = task.max_episode_steps
    config.random_process_fn = \
        lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.15, sigma=0.2,
                                         n_steps_annealing=100000)
    config.worker = DeterministicPolicyGradient
    config.min_memory_size = 50
    config.target_network_mix = 0.001
    config.test_interval = 0
    config.test_repetitions = 1
    config.gradient_clip = 40
    config.render_episode_freq = 0
    config.logger = Logger('./log', logger)
    run_episodes(DDPGAgent(config))

if __name__ == '__main__':
    mkdir('data')
    mkdir('data/video')
    mkdir('log')
    os.system('export OMP_NUM_THREADS=1')
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # dqn_cart_pole()
    # async_cart_pole()
    # a3c_cart_pole()
    # a3c_continuous()
    # p3o_continuous()
    # d3pg_continuous()
    ddpg_continuous()

    # dqn_fruit()
    # hrdqn_fruit()

    # dqn_pixel_atari('PongNoFrameskip-v4')
    # async_pixel_atari('PongNoFrameskip-v4')
    # a3c_pixel_atari('PongNoFrameskip-v4')

    # dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # async_pixel_atari('BreakoutNoFrameskip-v4')
    # a3c_pixel_atari('BreakoutNoFrameskip-v4')

    # acvp.train('PongNoFrameskip-v4')

