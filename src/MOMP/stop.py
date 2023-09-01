import numpy as np

class General:
    def __init__(self, maxIter=None, maxErr=None, maxRelErr=None):
        self.maxIter = maxIter
        self.maxErr = maxErr
        self.maxRelErr = maxRelErr
    def __call__(self, Y, Y_res, AX_I, *args, **kwargs):
        if self.maxIter is not None and AX_I.shape[-1] >= self.maxIter:
            return True
        Y_res_norm = np.linalg.norm(Y_res) / np.sqrt(Y_res.shape[-1])
        if self.maxErr is not None and Y_res_norm >= self.maxErr:
            return True
        Y_norm = np.linalg.norm(Y) / np.sqrt(Y.shape[-1])
        if self.maxRelErr is not None and Y_res_norm >= self.maxRelErr * Y_norm:
            return True
        return False
