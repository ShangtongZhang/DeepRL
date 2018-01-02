#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import gym.monitoring

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

        if config.render_episode_freq and ep % config.render_episode_freq == 0:
            video_recoder = gym.monitoring.VideoRecorder(
                env=agent.task.env, base_path='./data/video/%s-%s-%s-%d' % (agent_type, config.tag, agent.task.name, ep))
            agent.episode(True, video_recoder)
            video_recoder.close()

        if config.episode_limit and ep > config.episode_limit:
            break

        if config.max_steps and agent.total_steps > config.max_steps:
            break

        if config.test_interval and ep % config.test_interval == 0:
            config.logger.info('Testing...')
            agent.save('data/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name))
            test_rewards = []
            for _ in range(config.test_repetitions):
                test_rewards.append(agent.episode(True)[0])
            avg_reward = np.mean(test_rewards)
            avg_test_rewards.append(avg_reward)
            config.logger.info('Avg reward %f(%f)' % (
                avg_reward, np.std(test_rewards) / np.sqrt(config.test_repetitions)))
            with open('data/%s-%s-all-stats-%s.bin' % (agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards,
                             'steps': steps,
                             'test_rewards': avg_test_rewards}, f)
            if avg_reward > config.success_threshold:
                break

    return steps, rewards, avg_test_rewards

def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
