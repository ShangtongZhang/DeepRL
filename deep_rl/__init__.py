from .agent import *
from .component import *
from .network import *
from .utils import *
from gym.envs.registration import register

register(
    id='BairdPrediction-v0',
    entry_point='deep_rl.component.envs:BairdPrediction',
)

register(
    id='BairdControl-v0',
    entry_point='deep_rl.component.envs:BairdControl',
)
