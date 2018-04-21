#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self):
        return self.val

class LinearSchedule:
    def __init__(self, start, end, steps):
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self):
        val = self.current
        self.current = self.bound(self.current + self.inc, self.end)
        return val
