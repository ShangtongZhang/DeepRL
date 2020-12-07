from .agent import *
from .component import *
from .network import *
from .utils import *

from gym.envs.registration import register

register(
    id='RiskChain-v0',
    entry_point='deep_rl.component.envs:RiskChain',
)