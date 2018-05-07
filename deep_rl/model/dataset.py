#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

__all__ = ['generate_dataset']

from ..agent import *
from ..component import *
from deep_rl.utils import *
from skimage import io

def episode(agent, task):
    policy = GreedyPolicy(epsilon=0.2, final_step=1, min_epsilon=0.2)
    state_normalizer = ImageNormalizer()
    state = task.reset()
    total_rewards = 0.0
    steps = 0
    while True:
        state = np.stack([state_normalizer(state)])
        action_prob = agent.network.predict(state, True).flatten()
        action = policy.sample(action_prob)
        next_state, reward, done, _ = task.step(action)
        steps += 1
        total_rewards += reward
        state = next_state
        if done:
            break
    return total_rewards, steps

def generate_dataset(game, a2c_model, prefix):
    config = Config()
    config.history_length = 4
    config.num_workers = 1
    task_fn = lambda log_dir: PixelAtari(game, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda state_dim, action_dim: ActorCriticConvNet(
        config.history_length, action_dim, gpu=1)
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.97
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.logger = Logger('./log', logger, skip=True)
    agent = A2CAgent(config)
    agent.close()

    agent.load(a2c_model)
    task = PixelAtari(game, frame_skip=4, history_length=4, log_dir=None, dataset=True)

    ep = 0
    max_ep = 200
    mkdir('%s/dataset/%s' % (prefix, game))
    obs_sum = 0.0
    obs_count = 0
    while True:
        rewards, steps = episode(agent, task)
        path = '%s/dataset/%s/%05d' % (prefix, game, ep)
        mkdir(path)
        logger.info('Episode %d, reward %f, steps %d' % (ep, rewards, steps))
        with open('%s/action.bin' % (path), 'wb') as f:
            pickle.dump(task.dataset_env.saved_actions, f)
        obs_sum += np.asarray(task.dataset_env.saved_obs).sum(0)
        obs_count += len(task.dataset_env.saved_obs)
        for ind, obs in enumerate(task.dataset_env.saved_obs):
            io.imsave('%s/%05d.png' % (path, ind), obs)
        task.dataset_env.clear_saved()
        ep += 1
        if ep >= max_ep:
            break
    obs_mean = np.transpose(obs_sum, (2, 0, 1)) / obs_count
    with open('%s/dataset/%s/meta.bin' % (prefix, game), 'wb') as f:
        pickle.dump({'episodes': ep,
                     'mean_obs': obs_mean}, f)

