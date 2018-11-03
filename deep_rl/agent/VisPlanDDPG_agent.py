#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
from .PlanDDPG_agent import *

class VisPlanDDPGAgent(PlanDDPGAgent):
    def __init__(self, config):
        PlanDDPGAgent.__init__(self, config)

    def set_actor_index(self, actor_index):
        self.actor_index = actor_index

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        phi = self.network.feature(state)
        action = self.network.actor_models[self.actor_index](phi)
        action = action.cpu().detach().numpy()
        self.config.state_normalizer.unset_read_only()
        return action.flatten()

    def evaluation_episodes(self):
        rewards = []
        for ep in range(self.config.evaluation_episodes):
            rewards.append(self.deterministic_episode())
        self.config.logger.info('evaluation episode return: %f(%f)' % (
            np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))))
