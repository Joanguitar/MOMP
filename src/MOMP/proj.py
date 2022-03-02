import numpy as np
try:
    import cupy as cp
    bool_cp = True
    def vnorm(X):
        return cp.linalg.norm(cp.asarray(X), ord=2, axis=0).get()
    def vnorm2(X):
        return vnorm(X)**2
except:
    bool_cp = False
    def vnorm(X):
        return np.linalg.norm(X, ord=2, axis=0)
    def vnorm2(X):
        return np.sum(np.real(X*X.conj()), axis=0)

# OMP projection step
class OMP_proj:
    def __init__(self, X):
        self.X = X/vnorm(X)[np.newaxis]
    def __call__(self, Y_res):
        Y_res_X = np.dot(np.conj(Y_res).T, self.X)
        Y_res_X_norm = vnorm(Y_res_X)
        return np.argmax(Y_res_X_norm)

# MOMP projection initialization steps
class MOMP_greedy_proj:
    def __init__(self, A, X, sorting=None, normallized=True):
        self.A = A
        self.X = X
        self.normallized = normallized
        if sorting is None:
            sorting = [ii for ii in range(len(X))]
        self.sorting = sorting
        self.sorting_pre = sorting.copy()
        for ii_dim, sp in enumerate(sorting):
            for s in sorting[:ii_dim]:
                if s < sp:
                    self.sorting_pre[ii_dim] -= 1

    @profile
    def __call__(self, Y_res, YA=None, *args, **kwargs):
        ii = [None]*len(self.sorting)
        if self.normallized:
            AX = self.A.copy()
            for ii_dim, sp in enumerate(self.sorting_pre):
                AX = np.tensordot(AX, self.X[ii_dim], axes=(1+sp, 0))
                YAX = np.tensordot(np.conj(Y_res), AX, axes=(0, 0))
                if bool_cp:
                    AX_norm = vnorm(AX.reshape([-1, AX.shape[-1]]))
                    YAX_norm = vnorm(YAX.reshape([-1, YAX.shape[-1]]))
                    iii = np.argmax(YAX_norm/AX_norm)
                else:
                    AX_norm2 = vnorm2(AX.reshape([-1, AX.shape[-1]]))
                    YAX_norm2 = vnorm2(YAX.reshape([-1, YAX.shape[-1]]))
                    iii = np.argmax(YAX_norm2/AX_norm2)
                ii[ii_dim] = iii
                AX = AX[..., iii]
        else:
            if YA is None:
                YAX = np.tensordot(np.conj(Y_res), self.A, axes=(0, 0))
            else:
                YAX = YA.copy()
            for ii_dim, sp in enumerate(self.sorting_pre):
                YAX = np.tensordot(YAX, self.X[ii_dim], axes=(1+sp, 0))
                YAX_norm2 = vnorm2(YAX.reshape([-1, YAX.shape[-1]]))
                iii = np.argmax(YAX_norm2)
                ii[ii_dim] = iii
                YAX = YAX[..., iii]
        return ii


class MOMP_OMP_proj:
    def __init__(self, A, X_lr, X):
        self.A = A
        self.OMP_shape = [x.shape[1] for x in X_lr]
        X_OMP = A
        for x in X_lr:
            X_OMP = np.tensordot(X_OMP, x, axis=(1, 0))
        self.OMP_proj = OMP_proj(X_OMP)
        self.X = X
    def __call__(self, Y_res, *args, **kwargs):
        jj = np.ind2sub(self.OMP_proj(Y_res), self.OMP_shape)
        # MORE CODE

# MOMP projection step
class MOMP_proj:
    def __init__(self, A, X, n=2, initial=None, sorting=None, normallized=True):
        self.A = A
        self.X = X
        self.X_norm2 = [vnorm2(x) for x in X]
        self.n = n
        self.normallized = normallized
        if sorting is None:
            sorting = [ii for ii in range(len(X))]
        self.sorting = sorting
        if initial is None:
            initial = MOMP_greedy_proj(A, X, sorting=sorting, normallized=normallized)
        self.initial = initial
    @profile
    def __call__(self, Y_res):
        # Initial estimation
        if self.normallized:
            ii = self.initial(Y_res)
        else:
            YA = np.tensordot(np.conj(Y_res), self.A, axes=(0, 0))
            ii = self.initial(Y_res, YA=YA)
        # X_ii initialization
        X_ii = np.ones([x.shape[0] for x in self.X])
        for ii_dim, iii in enumerate(ii):
            X_ii = X_ii * self.X[ii_dim][:, iii].reshape([
                self.X[ii_dim].shape[0]
                if ii_dimp == ii_dim else 1
                for ii_dimp in range(len(self.X))])
        # Refinement iterations
        td_axis_r = [ii_dimp for ii_dimp in range(len(self.X)-1)]
        for _ in range(self.n):
            for ii_dim in self.sorting:
                iii = ii[ii_dim]
                td_axis_l = [
                    ii_dimp+1
                    for ii_dimp in range(len(self.X))
                    if ii_dimp != ii_dim]
                # Exclude (ii_dim, iii) from X_ii
                X_niii = np.tensordot(
                    X_ii, self.X[ii_dim][:, iii],
                    axes=(ii_dim, 0))/self.X_norm2[ii_dim][iii]
                # Compute projection
                if self.normallized:
                    AX_niii = np.tensordot(
                        self.A, X_niii,
                        axes=(td_axis_l, td_axis_r))
                    AX = np.dot(AX_niii, self.X[ii_dim])
                    YAX = np.tensordot(np.conj(Y_res), AX, axes=(0, 0))
                    AX_norm = vnorm(AX)
                    YAX_norm = vnorm(YAX)
                    iii = np.argmax(YAX_norm/AX_norm)
                else:
                    YAX_niii = np.tensordot(
                        YA, X_niii,
                        axes=(td_axis_l, td_axis_r))
                    YAX = np.dot(YAX_niii, self.X[ii_dim])
                    YAX_norm = vnorm(YAX)
                    iii = np.argmax(YAX_norm)
                ii[ii_dim] = iii
                # Update X_ii
                X_ii = np.expand_dims(X_niii, ii_dim) *\
                    self.X[ii_dim][:, iii].reshape([
                        self.X[ii_dim].shape[0]
                        if ii_dimp == ii_dim else 1
                        for ii_dimp in range(len(self.X))])
        AX_ii = np.tensordot(self.A, X_ii, axes=(
            [ii_dim+1 for ii_dim in range(len(self.sorting))],
            [ii_dim for ii_dim in range(len(self.sorting))]))
        return ii, AX_ii
