from async_agent import *
from dqn_agent import *

def async_cart_pole():
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.SGD(params, 0.001)
    config['network_fn'] = lambda: FullyConnectedNet([4, 50, 200, 2])
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, end_episode=500, min_epsilon=0.1)
    config['discount'] = 0.99
    config['target_network_update_freq'] = 200
    config['step_limit'] = 300
    config['n_workers'] = 8
    config['batch_size'] = 5
    config['test_interval'] = 500
    agent = AsyncAgent(**config)
    agent.run()

def dqn_cart_pole():
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.SGD(params, 0.001)
    config['network_fn'] = lambda optimizer_fn: FullyConnectedNet([4, 50, 200, 2], optimizer_fn)
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, end_episode=500, min_epsilon=0.1)
    config['replay_fn'] = lambda: Replay(memory_size=10000, batch_size=10)
    config['discount'] = 0.99
    config['target_network_update_freq'] = 200
    config['step_limit'] = 300
    agent = DQNAgent(**config)
    agent.run()

if __name__ == '__main__':
    async_cart_pole()
    # dqn_cart_pole()
