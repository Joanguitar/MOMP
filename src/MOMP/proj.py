import numpy as np
def vnorm(X):
    return np.linalg.norm(X, ord=2, axis=0)
def vnorm2(X):
    return np.sum(np.real(X*X.conj()), axis=0)

# OMP projection step
class OMP_proj:
    def __init__(self, X):
        self.X = X
        self.X_norm = X/vnorm(X)[np.newaxis]
    def __call__(self, Y_res):
        Y_res_X = np.dot(np.conj(Y_res).T, self.X_norm)
        Y_res_X_norm = vnorm(Y_res_X)
        ii = np.argmax(Y_res_X_norm)
        return ii, self.X[:, ii]

# Auxiliar functions for MOMP and SMOMP
def compute_X_ii(X_iii):
    sorting = np.argsort([x.shape[0] for x in X_iii])
    X_ii = 1
    for ii_dim in sorting:
        X_ii = X_ii * X_iii[ii_dim].reshape([
            X_iii[ii_dim].shape[0]
            if ii_dimp == ii_dim else 1
            for ii_dimp in range(len(X_iii))])
    return X_ii

def lr2hr(ii, Y, A, X, X_lr, sorting=None, normallized=True, YA=None):
    ii = ii.copy()
    if sorting is None:
        sorting = np.argsort([x.shape[0] for x in X])[::-1]
    X_iii = [x[:, iii] for x, iii in zip(X_lr, ii)]
    td_axis_r = [ii_dimp for ii_dimp in range(len(X)-1)]
    for ii_dim in sorting:
        iii = ii[ii_dim]
        td_axis_l = [
            ii_dimp+1
            for ii_dimp in range(len(X))
            if ii_dimp != ii_dim]
        # Exclude (ii_dim, iii) from X_ii
        X_niii = compute_X_ii([
            x for ii_dimp, x in enumerate(X_iii)
            if ii_dimp != ii_dim])
        # Compute projection
        if normallized:
            AX_niii = np.tensordot(
                A, X_niii,
                axes=(td_axis_l, td_axis_r))
            AX = np.dot(AX_niii, X[ii_dim])
            YAX = np.tensordot(np.conj(Y), AX, axes=(0, 0))
            AX_norm = vnorm(AX)
            YAX_norm = vnorm(YAX)
            iii = np.argmax(YAX_norm/AX_norm)
        else:
            if YA is None:
                YAX = np.tensordot(np.conj(Y), A, axes=(0, 0))
            YAX_niii = np.tensordot(
                YA, X_niii,
                axes=(td_axis_l, td_axis_r))
            YAX = np.dot(YAX_niii, X[ii_dim])
            YAX_norm = vnorm(YAX)
            iii = np.argmax(YAX_norm)
        ii[ii_dim] = iii
        # Update X_iii
        X_iii[ii_dim] = X[ii_dim][:, iii]
    return ii

# MOMP projection initialization steps
class MOMP_greedy_proj:
    def __init__(self, A, X, X_lr=None, sorting=None, normallized=True):
        self.A = A
        if X_lr is None:
            self.X = X
            self.X_hr = None
        else:
            self.X = X_lr
            self.X_hr = X
        self.normallized = normallized
        if sorting is None:
            sorting = np.argsort([x.shape[0] for x in self.X])[::-1]
        self.sorting = sorting
        self.sorting_pre = sorting.copy()
        for ii_dim, sp in enumerate(self.sorting):
            for s in self.sorting[:ii_dim]:
                if s < sp:
                    self.sorting_pre[ii_dim] -= 1

    def __call__(self, Y_res, YA=None, *args, **kwargs):
        ii = [None]*len(self.sorting)
        if self.normallized:
            AX = self.A.copy()
            for ii_dim, sp in zip(self.sorting, self.sorting_pre):
                AX = np.tensordot(AX, self.X[ii_dim], axes=(1+sp, 0))
                YAX = np.tensordot(np.conj(Y_res), AX, axes=(0, 0))
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
            for ii_dim, sp in zip(self.sorting, self.sorting_pre):
                YAX = np.tensordot(YAX, self.X[ii_dim], axes=(1+sp, 0))
                YAX_norm2 = vnorm2(YAX.reshape([-1, YAX.shape[-1]]))
                iii = np.argmax(YAX_norm2)
                ii[ii_dim] = iii
                YAX = YAX[..., iii]
        # Translate to high resolution
        if self.X_hr is not None:
            ii = lr2hr(ii, Y_res, self.A, self.X_hr, self.X, self.sorting, self.normallized, YA)
        return ii

class MOMP_OMP_proj:
    def __init__(self, A, X, X_lr, normallized=True):
        self.A = A
        self.OMP_shape = [x.shape[1] for x in X_lr]
        X_OMP = A
        for x in X_lr:
            X_OMP = np.tensordot(X_OMP, x, axis=(1, 0))
        self.OMP_proj = OMP_proj(X_OMP)
        self.X = X
        self.X_lr = X_lr
        self.normallized = normallized
    def __call__(self, Y_res, *args, **kwargs):
        ii = np.ind2sub(self.OMP_proj(Y_res), self.OMP_shape)
        ii = np.ind2sub(ii, self.OMP_shape)
        ii = lr2hr(ii, Y_res, self.A, self.X, self.X_lr, self.sorting, self.normallized)
        return ii

# MOMP projection step
class MOMP_proj:
    def __init__(self, A, X, n=2, initial=None, sorting=None, normallized=True):
        self.A = A
        self.X = X
        self.n = n
        self.normallized = normallized
        if sorting is None:
            sorting = np.argsort([x.shape[0] for x in self.X])[::-1]
        self.sorting = sorting
        if initial is None:
            initial = MOMP_greedy_proj(A, X, sorting=sorting, normallized=normallized)
        self.initial = initial

    def __call__(self, Y_res):
        # Initial estimation
        if self.normallized:
            ii = self.initial(Y_res)
        else:
            YA = np.tensordot(np.conj(Y_res), self.A, axes=(0, 0))
            ii = self.initial(Y_res, YA=YA)
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
                X_niii = compute_X_ii([
                    self.X[ii_dimp][:, iii]
                    for ii_dimp, iii in enumerate(ii)
                    if ii_dimp != ii_dim])
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
        # Compute X_ii
        X_ii = compute_X_ii([x[:, iii] for x, iii in zip(self.X, ii)])
        AX_ii = np.tensordot(self.A, X_ii, axes=(
            [ii_dim+1 for ii_dim in range(len(self.sorting))],
            [ii_dim for ii_dim in range(len(self.sorting))]))
        return ii, AX_ii

# SMOMP projection initialization step
class SMOMP_greedy_proj:
    def __init__(self, A, X, X_lr=None, sorting=None, normallized=True):
        self.A = A
        if X_lr is None:
            self.X = X
            self.X_hr = None
        else:
            self.X = X_lr
            self.X_hr = X
        self.normallized = normallized
        if sorting is None:
            sorting = np.argsort([x.shape[0] for x in self.X])[::-1]
        self.sorting = sorting
        self.sorting_pre = sorting.copy()
        for ii_dim, sp in enumerate(self.sorting):
            for s in self.sorting[:ii_dim]:
                if s < sp:
                    self.sorting_pre[ii_dim] -= 1

    def __call__(self, Y_res, YA=None, *args, **kwargs):
        ii = [None]*len(self.sorting)
        if self.normallized:
            AX = self.A.copy()
            for ii_dim, sp in zip(self.sorting, self.sorting_pre):
                AX = np.tensordot(AX, self.X[ii_dim], axes=(1+sp, 0))
                YAX = np.tensordot(np.conj(Y_res), AX, axes=(0, 0))
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
            for ii_dim, sp in zip(self.sorting, self.sorting_pre):
                YAX = np.tensordot(YAX, self.X[ii_dim], axes=(1+sp, 0))
                YAX_norm2 = vnorm2(YAX.reshape([-1, YAX.shape[-1]]))
                iii = np.argmax(YAX_norm2)
                ii[ii_dim] = iii
                YAX = YAX[..., iii]
        # Translate to high resolution
        if self.X_hr is not None:
            ii = lr2hr(ii, Y_res, self.A, self.X_hr, self.X, self.sorting, self.normallized, YA)
        return ii

# SMOMP projection step
class SMOMP_proj:
    def __init__(self, A, X, n=2, initial=None, sorting=None, normallized=True):
        self.A = []
        self.X_Ai = [None]*len(X)
        self.A_Xi = []
        self.Y_shape = []
        for a, Xi in A:
            if a is None:
                for ii_dim in Xi:
                    self.X_Ai[ii_dim] = len(self.A)
                    self.A.append(None)
                    self.A_Xi.append([ii_dim])
                    self.Y_shape.append(X[ii_dim].shape[0])
            else:
                for ii_dim in Xi:
                    self.X_Ai[ii_dim] = len(self.A)
                self.A.append(a)
                self.A_Xi.append(Xi)
                self.Y_shape.append(a.shape[0])
        self.X = X
        self.n = n
        self.normallized = normallized
        if sorting is None:
            sorting = np.argsort([x.shape[0] for x in self.X])[::-1]
        self.sorting = sorting
        if initial is None:
            initial = SMOMP_greedy_proj(A, X, sorting=sorting, normallized=normallized)
        self.initial = initial

    def __call__(self, Y_res):
        # Initial estimation
        if self.normallized:
            ii = self.initial(Y_res)
        else:
            YA = np.tensordot(np.conj(Y_res), self.A, axes=(0, 0))
            ii = self.initial(Y_res, YA=YA)
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
                X_niii = compute_X_ii([
                    self.X[ii_dimp][:, iii]
                    for ii_dimp, iii in enumerate(ii)
                    if ii_dimp != ii_dim])
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
        # Compute X_ii
        X_ii = compute_X_ii([x[:, iii] for x, iii in zip(self.X, ii)])
        AX_ii = np.tensordot(self.A, X_ii, axes=(
            [ii_dim+1 for ii_dim in range(len(self.sorting))],
            [ii_dim for ii_dim in range(len(self.sorting))]))
        return ii, AX_ii
