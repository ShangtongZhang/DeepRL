from .agent import *
from .component import *
from .network import *
from .utils import *

from gym.envs.registration import register

register(
    id='BoyansChainTabular-v0',
    entry_point='deep_rl.component.envs:BoyanChainTabular',
)

register(
    id='BoyansChainLinear-v0',
    entry_point='deep_rl.component.envs:BoyanChainLinear',
)