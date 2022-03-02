import numpy as np

class General:
    def __init__(self, maxIter):
        self.maxIter = maxIter
    def __call__(self, Y, Y_res, AX_I, *args, **kwargs):
        return AX_I.shape[-1] >= self.maxIter
