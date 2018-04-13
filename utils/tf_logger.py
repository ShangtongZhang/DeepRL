#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter
import os
import numpy as np
import torch

class Logger(object):
    def __init__(self, log_dir, vanilla_logger, skip=False):
        try:
            for f in os.listdir(log_dir):
                if not f.startswith('events'):
                    continue
                os.remove('%s/%s' % (log_dir, f))
        except IOError:
            os.mkdir(log_dir)
        if not skip:
            self.writer = SummaryWriter(log_dir)
        self.info = vanilla_logger.info
        self.debug = vanilla_logger.debug
        self.warning = vanilla_logger.warning
        self.skip = skip
        self.all_steps = {}

    def to_numpy(self, v):
        if isinstance(v, torch.autograd.Variable):
            v = v.data
        if isinstance(v, torch.FloatTensor):
            v = v.cpu().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def scalar_summary(self, tag, value, step=None):
        if self.skip:
            return
        self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def histo_summary(self, tag, values, step=None):
        if self.skip:
            return
        self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step, bins=1000)