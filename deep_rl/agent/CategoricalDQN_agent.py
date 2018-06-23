#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

class CategoricalDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.atoms = tensor(
            np.linspace(config.categorical_v_min,
                        config.categorical_v_max,
                        config.categorical_n_atoms))
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)

    def evaluation_action(self, state):
        value = self.network.predict(np.stack([self.config.state_normalizer(state)])).squeeze(0).detach()
        value = (value * self.atoms).sum(-1).cpu().detach().numpy().flatten()
        return np.argmax(value)

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while True:
            value = self.network.predict(np.stack([self.config.state_normalizer(state)])).squeeze(0).detach()
            # self.config.logger.histo_summary('prob', value, self.total_steps)
            value = (value * self.atoms).sum(-1).cpu().detach().numpy().flatten()
            # self.config.logger.histo_summary('q', value, self.total_steps)
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, _ = self.task.step(action)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1
            state = next_state

            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency == 0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                prob_next = self.target_network.predict(next_states).detach()
                q_next = (prob_next * self.atoms).sum(-1)
                # self.config.logger.histo_summary('q next', q_next.cpu().detach().numpy(), self.total_steps)
                _, a_next = torch.max(q_next, dim=1)
                a_next = a_next.view(-1, 1, 1).expand(-1, -1, prob_next.size(2))
                prob_next = prob_next.gather(1, a_next).squeeze(1)
                # self.config.logger.histo_summary('prob next', prob_next.cpu().detach().numpy(), self.total_steps)

                rewards = tensor(rewards)
                terminals = tensor(terminals)
                atoms_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * self.atoms.view(1, -1)
                # epsilon = 1e-5
                atoms_next.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
                b = (atoms_next - self.config.categorical_v_min) / self.delta_atom
                l = b.floor()
                u = b.ceil()
                d_m_l = (u + (l == u).float() - b) * prob_next
                d_m_u = (b - l) * prob_next
                target_prob = tensor(np.zeros(prob_next.size()))
                for i in range(target_prob.size(0)):
                    target_prob[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_prob[i].index_add_(0, u[i].long(), d_m_u[i])

                prob = self.network.predict(states)
                actions = tensor(actions).long()
                actions = actions.view(-1, 1, 1).expand(-1, -1, prob.size(2))
                prob = prob.gather(1, actions).squeeze(1)
                loss = -(target_prob * prob.log()).sum(-1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps
