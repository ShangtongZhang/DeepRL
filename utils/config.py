#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

class Config:
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.policy_fn = None
        self.replay_fn = None
        self.discount = 0.99
        self.target_network_update_freq = 0
        self.max_episode_length = 0
        self.exploration_steps = 0
        self.logger = None
        self.history_length = 1
        self.test_interval = 100
        self.test_repetitions = 50
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.worker = None
        self.update_interval = 1
        self.gradient_clip = 40
        self.entropy_weight = 0.01
        self.gae_tau = 1.0
        self.reward_shift_fn = lambda r: r
        self.state_shift_fn = lambda s: s
        self.action_shift_fn = lambda a: a
