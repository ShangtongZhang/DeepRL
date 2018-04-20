#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .normalizer import *

class Config:
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.policy_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.discount = 0.99
        self.target_network_update_freq = 0
        self.max_episode_length = 0
        self.exploration_steps = 0
        self.logger = None
        self.history_length = 1
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.update_interval = 1
        self.gradient_clip = 0.5
        self.entropy_weight = 0.01
        self.use_gae = False
        self.gae_tau = 1.0
        self.noise_decay_interval = 0
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.hybrid_reward = False
        self.episode_limit = 0
        self.min_memory_size = 200
        self.master_fn = None
        self.master_optimizer_fn = None
        self.num_heads = 10
        self.min_epsilon = 0
        self.save_interval = 0
        self.max_steps = 0
        self.success_threshold = float('inf')
        self.render_episode_freq = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
        self.categorical_v_min = -10
        self.categorical_v_max = 10
        self.categorical_n_atoms = 51
        self.num_quantiles = 10
        self.gaussian_noise_scale = 0.3
        self.optimization_epochs = 4
        self.num_mini_batches = 32
        self.test_interval = 0
        self.test_repetitions = 10
        self.evaluation_env = None
