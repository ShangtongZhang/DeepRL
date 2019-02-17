#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class OffPACAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

        self.episode_rewards = []
        self.online_rewards = np.zeros(config.num_workers)

    def random_action(self):
        config = self.config
        return np.random.randint(0, config.action_dim, size=(config.num_workers, ))

    def update(self, s, a, mu_a, r, next_s, m):
        config = self.config
        prediction = self.network(s, a)
        with torch.no_grad():
            target = self.network(next_s)['v']
            target = r + config.discount * m * target
        td_error = target - prediction['v']
        v_loss = td_error.pow(2).mul(0.5).mean()
        rho = prediction['pi_a'] / mu_a
        entropy = prediction['ent'].mean()
        pi_loss = -rho.detach() * td_error.detach() * prediction['log_pi_a'] - config.entropy_weight * entropy
        pi_loss = pi_loss.mean()

        loss = (v_loss + pi_loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

    def eval_step(self, state):
        with torch.no_grad():
            action = self.network(state)['a']
        return np.asscalar(to_np(action))


    def step(self):
        config = self.config
        actions = self.random_action()
        mu_a = np.zeros_like(actions) + 1 / config.action_dim
        next_states, rewards, terminals, _ = self.task.step(actions)
        self.online_rewards += rewards
        rewards = config.reward_normalizer(rewards)
        for i, terminal in enumerate(terminals):
            if terminals[i]:
                self.episode_rewards.append(self.online_rewards[i])
                self.online_rewards[i] = 0

        mask = (1 - terminals).astype(np.uint8)
        self.update(
            tensor(self.states),
            tensor(actions),
            tensor(mu_a).unsqueeze(1),
            tensor(rewards).unsqueeze(1),
            tensor(next_states),
            tensor(mask).unsqueeze(1))
        self.states = next_states

        self.total_steps += 1
