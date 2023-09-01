import numpy as np

class General:
    def __init__(self, maxIter=None, minErr=None, minRelErr=None):
        self.maxIter = maxIter
        self.minErr = minErr
        self.minRelErr = minRelErr
    def __call__(self, Y, Y_res, AX_I, *args, **kwargs):
        if self.maxIter is not None and AX_I.shape[-1] >= self.maxIter:
            return True
        Y_res_norm = np.linalg.norm(Y_res) / np.sqrt(Y_res.shape[-1])
        if self.minErr is not None and Y_res_norm <= self.minErr:
            return True
        Y_norm = np.linalg.norm(Y) / np.sqrt(Y.shape[-1])
        if self.minRelErr is not None and Y_res_norm <= self.minRelErr * Y_norm:
            return True
        return False
