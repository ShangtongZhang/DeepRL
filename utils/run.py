#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle


def run_episodes(agent):
    config = agent.config
    window_size = 100
    ep = 0
    rewards = []
    steps = []
    avg_test_rewards = []
    agent_type = agent.__class__.__name__
    while True:
        ep += 1
        reward, step = agent.episode()
        rewards.append(reward)
        steps.append(step)
        avg_reward = np.mean(rewards[-window_size:])
        config.logger.info('episode %d, reward %f, avg reward %f, total steps %d, episode step %d' % (
            ep, reward, avg_reward, agent.total_steps, step))

        if config.save_interval and ep % config.save_interval == 0:
            with open('data/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump([steps, rewards], f)

        if config.episode_limit and ep > config.episode_limit:
            break

        if config.test_interval and ep % config.test_interval == 0:
            config.logger.info('Testing...')
            agent.save('data/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name))
            test_rewards = []
            for _ in range(config.test_repetitions):
                test_rewards.append(agent.episode(True))
            avg_reward = np.mean(test_rewards)
            avg_test_rewards.append(avg_reward)
            config.logger.info('Avg reward %f(%f)' % (
                avg_reward, np.std(test_rewards) / np.sqrt(config.test_repetitions)))
            with open('data/%s-%s-all-stats-%s.bin' % (agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards,
                             'steps': steps,
                             'test_rewards': avg_test_rewards}, f)
            if avg_reward > agent.task.success_threshold:
                break

    return steps, rewards, avg_test_rewards