from .agent import *
from .component import *
from .network import *
from .utils import *

from gym.envs.registration import register

register(
    id='RobotTabular-v0',
    entry_point='deep_rl.component.envs:RobotTabular',
)

register(
    id='RobotLinear-v0',
    entry_point='deep_rl.component.envs:RobotLinear',
)

register(
    id='Reacher-v3',
    entry_point='deep_rl.component.envs:ReacherFixed',
    max_episode_steps=50,
)