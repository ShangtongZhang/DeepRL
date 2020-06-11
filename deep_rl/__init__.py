from .agent import *
from .component import *
from .network import *
from .utils import *

from gym.envs.registration import register

register(
    id='OriginalBaird-v0',
    entry_point='deep_rl.component.envs:OriginalBaird',
)

register(
    id='OneHotBaird-v0',
    entry_point='deep_rl.component.envs:OneHotBaird',
)

register(
    id='ZeroHotBaird-v0',
    entry_point='deep_rl.component.envs:ZeroHotBaird',
)

register(
    id='AliasedBaird-v0',
    entry_point='deep_rl.component.envs:AliasedBaird',
)
