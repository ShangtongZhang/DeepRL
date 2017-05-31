#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np

def NStepQLearning(batch_states, batch_actions, batch_rewards,
                   tailing_state, tailing_action, terminal, network, discount):
    if terminal:
        reward = 0
    else:
        reward = np.max(network.predict(np.stack([tailing_state])).flatten())
    rewards = []
    for r in reversed(batch_rewards):
        reward = r + discount * reward
        rewards.append(reward)
    return rewards

def OneStepQLearning(batch_states, batch_actions, batch_rewards,
                     tailing_state, tailing_action, terminal, network, discount):
    batch_states.append(tailing_state)
    q_next = network.predict(np.asarray(batch_states[1:]))
    q_next = np.max(q_next, axis=1)
    if terminal:
        q_next[-1] = 0
    batch_states.pop(-1)
    batch_rewards = np.asarray(batch_rewards) + discount * q_next
    return batch_rewards

def OneStepSarsa(batch_states, batch_actions, batch_rewards,
                 tailing_state, tailing_action, terminal, network, discount):
    batch_states.append(tailing_state)
    batch_actions.append(tailing_action)
    q_next = network.predict(np.asarray(batch_states[1:]))
    q_next = q_next[np.arange(len(batch_actions[1:])), batch_actions[1:]]
    if terminal:
        q_next[-1] = 0
    batch_states.pop(-1)
    batch_actions.pop(-1)
    batch_rewards = np.asarray(batch_rewards) + discount * q_next
    return batch_rewards

def AdvantageActorCritic(batch_states, batch_actions, batch_rewards,
                         tailing_state, tailing_action, terminal, network, discount):
    if terminal:
        reward = 0
    else:
        reward = np.asscalar(network.critic(np.stack([tailing_state])))
    rewards = []
    for r in reversed(batch_rewards):
        reward = r + discount * reward
        rewards.append(reward)
    return rewards
