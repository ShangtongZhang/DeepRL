#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter
import os
import numpy as np

class Logger(object):
    def __init__(self, log_dir, vanilla_logger, skip=False):
        try:
            for f in os.listdir(log_dir):
                os.remove('%s/%s' % (log_dir, f))
        except IOError:
            os.mkdir(log_dir)
        if not skip:
            self.writer = SummaryWriter(log_dir)
        self.info = vanilla_logger.info
        self.debug = vanilla_logger.debug
        self.warning = vanilla_logger.warning
        self.skip = skip
        self.step = 0

    def scalar_summary(self, tag, value, step=None):
        if self.skip:
            return
        if step is None:
            step = self.step
            self.step += 1
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def histo_summary(self, tag, values, step=None):
        if self.skip:
            return
        if step is None:
            step = self.step
            self.step += 1
        self.writer.add_histogram(tag, values, step, bins=1000)