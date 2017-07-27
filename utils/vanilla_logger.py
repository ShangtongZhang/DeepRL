import numpy as np

class Logger(object):
    def __init__(self, log_dir, vanilla_logger, skip=False):
        """Create a summary writer logging to log_dir."""
        self.info = vanilla_logger.info
        self.debug = vanilla_logger.debug
        self.warning = vanilla_logger.warning
        self.skip = skip

    def scalar_summary(self, tag, value, step):
        if self.skip:
            return

    def image_summary(self, tag, images, step):
        if self.skip:
            return

    def histo_summary(self, tag, values, step, bins=1000):
        if self.skip:
            return
