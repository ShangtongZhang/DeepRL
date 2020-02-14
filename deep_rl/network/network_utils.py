#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import *


class BaseNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.is_recur = False
        self.config = config


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def lstm_init(lstm, w_scale=1.0):
    for layer_p in lstm._all_weights:
        for p in layer_p:
            if 'weight' in p:
                nn.init.orthogonal_(lstm.__getattr__(p).data)
                lstm.__getattr__(p).data.mul_(w_scale)
            if 'bias' in p:
                nn.init.constant_(lstm.__getattr__(p).data, 0)
    return lstm
