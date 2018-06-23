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
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())
        self.policy = config.policy_fn()

        self.total_steps = 0
        self.states = self.task.reset()
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            q = self.network.predict(self.config.state_normalizer(states))
            actions = [self.policy.sample(v) for v in q.cpu().detach().numpy()]
            next_states, rewards, terminals, _ = self.task.step(actions)
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([q, actions, rewards, 1 - terminals])
            states = next_states

            self.policy.update_epsilon(config.num_workers)
            self.total_steps += config.num_workers
            if self.total_steps / config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        processed_rollout = [None] * (len(rollout))
        returns = self.target_network.predict(config.state_normalizer(states)).detach()
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

        self.evaluate(config.rollout_length)