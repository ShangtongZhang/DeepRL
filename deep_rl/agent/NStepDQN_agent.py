#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *

class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []

    def step(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            q = self.network(self.config.state_normalizer(states))

            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q))

            next_states, rewards, terminals, _ = self.task.step(actions)
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0

            rollout.append([q, actions, rewards, 1 - terminals])
            states = next_states

            self.total_steps += config.num_workers
            if self.total_steps / config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        processed_rollout = [None] * (len(rollout))
        returns = self.target_network(config.state_normalizer(states)).detach()
        returns, _ = torch.max(returns, dim=1, keepdim=True)
        for i in reversed(range(len(rollout))):
            q, actions, rewards, terminals = rollout[i]
            actions = tensor(actions).unsqueeze(1).long()
            q = q.gather(1, actions)
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns
            processed_rollout[i] = [q, returns]

        q, returns= map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        loss = 0.5 * (q - returns).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
