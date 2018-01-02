#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter

class Logger(object):
    def __init__(self, log_dir, vanilla_logger, skip=False):
        self.writer = SummaryWriter(log_dir)
        self.info = vanilla_logger.info
        self.debug = vanilla_logger.debug
        self.warning = vanilla_logger.warning
        self.skip = skip

    def scalar_summary(self, tag, value, step):
        if self.skip:
            return
        self.writer.add_scalar(tag, value, step)

    def histo_summary(self, tag, values, step):
        if self.skip:
            return
        self.writer.add_histogram(tag, values, step, bins=1000)