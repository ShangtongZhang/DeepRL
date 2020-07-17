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


class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def compute_q(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        if config.noisy_linear:
            self._network.reset_noise()
        with config.lock:
            prediction = self._network(config.state_normalizer(self._state))
        q_values = self.compute_q(prediction)

        if config.noisy_linear:
            epsilon = 0
        elif self._total_steps < config.exploration_steps:
            epsilon = 1
        else:
            epsilon = config.random_action_prob()
        action = epsilon_greedy(epsilon, q_values)
        next_state, reward, done, info = self._task.step(action)
        entry = [self._state, action, reward, next_state, done, info]
        self._total_steps += 1
        self._state = next_state
        return entry


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)
        self.total_steps = 0

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        q_target = rewards + self.config.discount ** config.n_step * q_next * masks
        actions = tensor(transitions.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def step(self):
        config = self.config
        transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()
            if config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()
            loss = self.compute_loss(transitions)
            if isinstance(transitions, PrioritizedTransition):
                priorities = loss.abs().add(config.replay_eps).pow(config.replay_alpha)
                idxs = tensor(transitions.idx).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(transitions.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
                weights = weights / weights.max()
                loss = loss.mul(weights)

            loss = self.reduce_loss(loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
