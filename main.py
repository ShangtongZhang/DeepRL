from async_agent import *
from dqn_agent import *

def async_cart_pole():
    config = dict()
    config['task_fn'] = lambda: CartPole()
    config['optimizer_fn'] = lambda params: torch.optim.SGD(params, 0.001)
    config['network_fn'] = lambda: FullyConnectedNet([4, 50, 200, 2])
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, end_episode=500, min_epsilon=0.1)
    # config['bootstrap_fn'] = OneStepQLearning
    # config['bootstrap_fn'] = NStepQLearning
    config['bootstrap_fn'] = OneStepSarsa
    config['discount'] = 0.99
    config['target_network_update_freq'] = 200
    config['step_limit'] = 300
    config['n_workers'] = 8
    config['batch_size'] = 5
    config['test_interval'] = 500
    config['test_repeats'] = 5
    agent = AsyncAgent(**config)
    agent.run()

def async_lunar_lander():
    config = dict()
    config['task_fn'] = lambda: LunarLander()
    config['optimizer_fn'] = lambda params: torch.optim.Adam(params, 0.001)
    config['network_fn'] = lambda: FullyConnectedNet([8, 50, 200, 4])
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=1.0, end_episode=2000, min_epsilon=0.05)
    config['bootstrap_fn'] = OneStepQLearning
    config['discount'] = 0.99
    config['target_network_update_freq'] = 200
    config['step_limit'] = 5000
    config['n_workers'] = 8
    config['batch_size'] = 10
    config['test_interval'] = 1000
    config['test_repeats'] = 5
    agent = AsyncAgent(**config)
    agent.run()

# Mountain Car is fairly unstable
def dqn_mountain_car():
    config = dict()
    config['task_fn'] = lambda: MountainCar()
    config['optimizer_fn'] = lambda params: torch.optim.SGD(params, 0.001)
    config['network_fn'] = lambda optimizer_fn: FullyConnectedNet([2, 50, 200, 3], optimizer_fn)
    config['policy_fn'] = lambda: GreedyPolicy(epsilon=0.5, end_episode=500, min_epsilon=0.1)
    config['replay_fn'] = lambda: Replay(memory_size=10000, batch_size=10)
    config['discount'] = 0.99
    config['target_network_update_freq'] = 1000
    config['step_limit'] = 5000
    agent = DQNAgent(**config)
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
    # async_lunar_lander()
    # dqn_cart_pole()
    # dqn_mountain_car()
