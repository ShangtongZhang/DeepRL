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


class RainbowActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _set_up(self):
        self.config.atoms = tensor(self.config.atoms)

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with torch.no_grad():
            with config.lock:
                self._network.reset_noise()
                probs, _ = self._network(config.state_normalizer(self._state))
        q_values = (probs * self.config.atoms).sum(-1)
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class RainbowAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()
        config.atoms = np.linspace(config.categorical_v_min,
                                   config.categorical_v_max, config.categorical_n_atoms)

        self.replay = config.replay_fn()
        self.actor = RainbowActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.n_step_cache = deque(maxlen=config.n_step)
        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)
        self.atoms = tensor(config.atoms)
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)
        self.n_step_cache = []

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prob, _ = self.network(state)
        q = (prob * self.atoms).sum(-1)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return [action]

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            self.n_step_cache.append([state, action, reward, next_state, done])
            if len(self.n_step_cache) == config.n_step:
                cum_r = 0
                cum_done = 0
                for s, a, r, _, done in reversed(self.n_step_cache):
                    cum_r = r + (1 - done) * config.discount * cum_r
                    cum_done = done or cum_done
                experiences.append([s, a, cum_r, next_state, cum_done])
                self.n_step_cache.pop(0)
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            if config.replay_type == Config.PRIORITIZED_REPLAY:
                states, actions, rewards, next_states, terminals, sampling_probs, idxs = experiences
            elif config.replay_type == Config.DEFAULT_REPLAY:
                states, actions, rewards, next_states, terminals = experiences

            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)

            self.target_network.reset_noise()
            self.network.reset_noise()
            with torch.no_grad():
                prob_next, _ = self.target_network(next_states)
                q_next = (prob_next * self.atoms).sum(-1)
                if config.double_q:
                    a_next = torch.argmax((self.network(next_states)[0] * self.atoms).sum(-1), dim=-1)
                else:
                    a_next = torch.argmax(q_next, dim=-1)
                prob_next = prob_next[self.batch_indices, a_next, :]

            rewards = tensor(rewards).unsqueeze(-1)
            terminals = tensor(terminals).unsqueeze(-1)
            atoms_target = rewards + self.config.discount ** config.n_step * (1 - terminals) * self.atoms.view(1, -1)
            atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
            atoms_target = atoms_target.unsqueeze(1)
            target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                          prob_next.unsqueeze(1)
            target_prob = target_prob.sum(-1)

            _, log_prob = self.network(states)
            actions = tensor(actions).long()
            log_prob = log_prob[self.batch_indices, actions, :]
            KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
            priorities = KL.abs().add(config.replay_eps).pow(config.replay_alpha)
            if config.replay_type == Config.PRIORITIZED_REPLAY:
                idxs = tensor(idxs).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(sampling_probs)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
                weights = weights / weights.max()
                KL = KL.mul(weights)

            loss = KL.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
