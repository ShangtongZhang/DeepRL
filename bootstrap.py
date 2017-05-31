#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np

def NStepQLearning(batch_states, batch_actions, batch_rewards,
                   tailing_state, tailing_action, terminal, agent):
    if terminal:
        reward = 0
    else:
        with agent.network_lock:
            reward = np.max(agent.target_network.predict(
                np.reshape(tailing_state, (1, ) + tailing_state.shape)))
    rewards = []
    for r in reversed(batch_rewards):
        reward = r + agent.discount * reward
        rewards.append(reward)
    return rewards

def OneStepQLearning(batch_states, batch_actions, batch_rewards,
                     tailing_state, tailing_action, terminal, agent):
    batch_states.append(tailing_state)
    with agent.network_lock:
        q_next = agent.target_network.predict(np.asarray(batch_states[1:]))
    q_next = np.max(q_next, axis=1)
    if terminal:
        q_next[-1] = 0
    batch_states.pop(-1)
    batch_rewards = np.asarray(batch_rewards) + agent.discount * q_next
    return batch_rewards

def OneStepSarsa(batch_states, batch_actions, batch_rewards,
                 tailing_state, tailing_action, terminal, agent):
    batch_states.append(tailing_state)
    batch_actions.append(tailing_action)
    with agent.network_lock:
        q_next = agent.target_network.predict(np.asarray(batch_states[1:]))
    q_next = q_next[np.arange(len(batch_actions[1:])), batch_actions[1:]]
    if terminal:
        q_next[-1] = 0
    batch_states.pop(-1)
    batch_actions.pop(-1)
    batch_rewards = np.asarray(batch_rewards) + agent.discount * q_next
    return batch_rewards

def AdvantageActorCritic(batch_states, batch_actions, batch_rewards,
                         tailing_state, tailing_action, terminal, agent):
    if terminal:
        reward = 0
    else:
        with agent.network_lock:
            reward = np.asscalar(agent.learning_network.critic(np.stack([tailing_state])))
    rewards = []
    for r in reversed(batch_rewards):
        reward = r + agent.discount * reward
        rewards.append(reward)
    return rewards
