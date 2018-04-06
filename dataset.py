#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from agent import *
from component import *
from utils import *
import torchvision
import torch
from skimage import io

# PREFIX = '.'
PREFIX = '/local/data'

def dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda state_dim, action_dim: ConvNet(config.history_length, action_dim, gpu=0)
    # config.network_fn = lambda state_dim, action_dim: DuelingConvNet(config.history_length, action_dim)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=32, dtype=np.uint8)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = Logger('./log', logger)
    # config.double_q = True
    config.double_q = False
    return DQNAgent(config)

def train_dqn(game):
    agent = dqn_pixel_atari(game)
    run_episodes(agent)

def episode(env, agent):
    config = agent.config
    policy = GreedyPolicy(epsilon=0.3, final_step=1, min_epsilon=0.3)
    state = env.reset()
    history_buffer = [state] * config.history_length
    state = np.vstack(history_buffer)
    total_reward = 0.0
    steps = 0
    while True:
        value = agent.network.predict(np.stack([state]), False)
        value = value.cpu().data.numpy().flatten()
        action = policy.sample(value)
        next_state, reward, done, info = env.step(action)
        history_buffer.pop(0)
        history_buffer.append(next_state)
        state = np.vstack(history_buffer)
        done = (done or (config.max_episode_length and steps > config.max_episode_length))
        steps += 1
        total_reward += reward
        if done:
            break
    return total_reward, steps

def generate_dateset(game):
    agent = dqn_pixel_atari(game)
    model_file = 'data/%s-%s-model-%s.bin' % (agent.__class__.__name__, agent.config.tag, agent.task.name)
    agent.load(model_file)

    env = make_atari(game, frame_skip=4)
    env = EpisodicLifeEnv(env)
    dataset_env = DatasetEnv(env)
    env = wrap_deepmind(env, history_length=4)

    ep = 0
    max_ep = 200
    mkdir('%s/dataset/%s' % (PREFIX, game))
    obs_sum = 0.0
    obs_count = 0
    while True:
        rewards, steps = episode(env, agent)
        path = '%s/dataset/%s/%05d' % (PREFIX, game, ep)
        mkdir(path)
        logger.info('Episode %d, reward %f, steps %d' % (ep, rewards, steps))
        with open('%s/action.bin' % (path), 'wb') as f:
            pickle.dump(dataset_env.saved_actions, f)
        obs_sum += np.asarray(dataset_env.saved_obs).sum(0)
        obs_count += len(dataset_env.saved_obs)
        for ind, obs in enumerate(dataset_env.saved_obs):
            io.imsave('%s/%05d.png' % (path, ind), obs)
        dataset_env.clear_saved()
        ep += 1
        if ep >= max_ep:
            break
    obs_mean = np.transpose(obs_sum, (2, 0, 1)) / obs_count
    with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'wb') as f:
        pickle.dump({'episodes': ep,
                     'mean_obs': obs_mean}, f)

if __name__ == '__main__':
    mkdir('dataset')
    game = 'PongNoFrameskip-v4'
    # train_dqn(game)
    generate_dateset(game)