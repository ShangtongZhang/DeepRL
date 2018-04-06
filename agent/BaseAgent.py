#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch

class BaseAgent:
    def __init__(self):
        pass

    def close(self):
        if hasattr(self.task, 'close'):
            self.task.close()

    def save(self, filename):
        pass
