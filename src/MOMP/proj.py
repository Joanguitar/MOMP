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

class OMP_proj_FT:
    def __init__(self, steps=3, res=7):
        self.steps = steps
        if res < 5:
            raise ValueError('res must be at least 5')
        self.res = res
        self.theta_dom_inital = []
    def build(self, Y_res):
        theta_dom = np.linspace(0, 2*np.pi, 2*len(Y_res)+1)[:-1]
        self.theta_dom_inital = theta_dom.copy()
        self.X_initial = np.exp(1j * theta_dom[None] * np.arange(len(Y_res))[:, None])
        theta_delta = np.pi / len(Y_res)
        self.theta_dom = []
        self.X = []
        for _ in range(self.steps):
            theta_delta *= (2 - 1 / (self.res + 1)) / (self.res - 1)
            theta_dom = np.linspace(-theta_delta, theta_delta, self.res)
            self.theta_dom.append(theta_dom.copy())
            self.X.append(np.exp(1j * theta_dom[None] * np.arange(len(Y_res))[:, None]))
    def __call__(self, Y_res):
        if len(self.theta_dom_inital) != 2 * len(Y_res):
            print("BUILD")
            self.build(Y_res)
        Y_res_X = np.dot(np.conj(Y_res).T, self.X_initial)
        Y_res_X_norm = vnorm(Y_res_X)
        ii = np.argmax(Y_res_X_norm)
        theta = self.theta_dom_inital[ii]
        for theta_dom, X in zip(self.theta_dom, self.X):
            Y_res_X = np.dot(np.conj(Y_res).T, np.exp(1j * theta * np.arange(len(Y_res))[:, None]) * X)
            Y_res_X_norm = vnorm(Y_res_X)
            ii = np.argmax(Y_res_X_norm)
            theta += theta_dom[ii]
        return theta, np.exp(1j * theta * np.arange(len(Y_res)))

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
                YA = np.tensordot(np.conj(Y), A, axes=(0, 0))
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

def lr2hr_S(ii, Y, A, X, X_lr, sorting=None, normallized=True, YA=None):
    ii = ii.copy()
    if sorting is None:
        sorting = np.argsort([x.shape[0] for x in X])[::-1]
    X_Ai = [None]*len(X)
    Y_shape = []
    ii_dim = -1
    for ii_a, a, in enumerate(A):
        if a is None:
            ii_dim += 1
            X_Ai[ii_dim] = ii_a
            Y_shape.append(X[ii_dim].shape[0])
        else:
            for _ in range(len(a.shape)-1):
                ii_dim += 1
                X_Ai[ii_dim] = ii_a
            Y_shape.append(a.shape[0])
    X_iii = [x[:, iii] for x, iii in zip(X_lr, ii)]
    if normallized:
        # Compute AX_ii
        X_ii = [
            compute_X_ii([
                x
                for ii_dim, x in enumerate(X_iii)
                if X_Ai[ii_dim] == ii_a])
            for ii_a, a in enumerate(A)]
        AX_ii = [
            np.tensordot(a, x_ii, axes=(
                [ii_dim for ii_dim in range(1, len(a.shape))],
                [ii_dim for ii_dim in range(len(x_ii.shape))]))
            if a is not None else x_ii
            for a, x_ii in zip(A, X_ii)]
    else:
        if YA is None:
            YA = Y.conj().reshape(Y_shape+[-1])
            for a in A:
                if a is not None:
                    YA = np.tensordot(YA, a, axes=(0, 0))
                else:
                    YA = YA.transpose([ii for ii in range(1, len(YA.shape))]+[0])
    td_axis_r = [ii_dimp for ii_dimp in range(len(X)-1)]
    for ii_dim in sorting:
        iii = ii[ii_dim]
        td_axis_l = [
            ii_dimp+1
            for ii_dimp in range(len(X))
            if ii_dimp != ii_dim]
        # Compute projection
        if normallized:
            # Exclude (ii_dim, iii) from X_ii[self.X_Ai[ii_dim]]
            x_niii = compute_X_ii([
                x
                for ii_dimp, x in enumerate(X_iii)
                if ii_dimp != ii_dim
                and X_Ai[ii_dimp] == X_Ai[ii_dim]])
            A_Xi = [
                ii_dimp for ii_dimp, Ai in enumerate(X_Ai)
                if Ai == X_Ai[ii_dim]]
            td_axis_l = [
                ii_dimp_ii+1 for ii_dimp_ii, ii_dimp in enumerate(A_Xi)
                if ii_dimp != ii_dim]
            td_axis_r = [ii_dimp for ii_dimp in range(len(td_axis_l))]
            if A[X_Ai[ii_dim]] is not None:
                ax_niii = np.tensordot(
                    A[X_Ai[ii_dim]], x_niii,
                    axes=(td_axis_l, td_axis_r))
                ax = np.dot(ax_niii, X[ii_dim])
            else:
                ax = X[ii_dim]
            YAX_nXiidim = np.tensordot(
                Y.conj().reshape(Y_shape+[-1]),
                compute_X_ii([
                    ax for ii_a, ax in enumerate(AX_ii)
                    if ii_a != X_Ai[ii_dim]]),
                axes=(
                    [
                        ii_a for ii_a in range(len(A))
                        if ii_a != X_Ai[ii_dim]],
                    [ii_a for ii_a in range(len(A)-1)]))
            YAX = np.tensordot(YAX_nXiidim, ax, axes=(0, 0))
            AX_norm = vnorm(ax)
            YAX_norm = vnorm(YAX)
            iii = np.argmax(YAX_norm/AX_norm)
        else:
            # Exclude (ii_dim, iii) from X_ii
            X_niii = compute_X_ii([
                x for ii_dimp, x in enumerate(X_iii)
                if ii_dimp != ii_dim])
            YAX_niii = np.tensordot(
                YA, X_niii,
                axes=(td_axis_l, td_axis_r))
            YAX = np.dot(YAX_niii, X[ii_dim])
            YAX_norm = vnorm(YAX)
            iii = np.argmax(YAX_norm)
        ii[ii_dim] = iii
        # Update X_iii
        X_iii[ii_dim] = X[ii_dim][:, iii]
        if normallized:
            X_ii[X_Ai[ii_dim]] = compute_X_ii([
                x
                for ii_dimp, x in enumerate(X_iii)
                if X_Ai[ii_dimp] == X_Ai[ii_dim]])
            AX_ii[X_Ai[ii_dim]] = ax[:, iii]
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
        self.X_Ai = [None]*len(X)
        self.Y_shape = []
        ii_dim = -1
        for ii_a, a, in enumerate(A):
            if a is None:
                ii_dim += 1
                self.X_Ai[ii_dim] = ii_a
                self.Y_shape.append(self.X[ii_dim].shape[0])
            else:
                for _ in range(len(a.shape)-1):
                    ii_dim += 1
                    self.X_Ai[ii_dim] = ii_a
                self.Y_shape.append(a.shape[0])
        self.normallized = normallized
        if sorting is None:
            sorting = np.argsort([x.shape[0] for x in self.X])[::-1]
        self.sorting = sorting
        self.sorting_pre = sorting.copy()
        for ii_dim, sp in enumerate(self.sorting):
            for s in self.sorting[:ii_dim]:
                if s < sp:
                    self.sorting_pre[ii_dim] -= 1
        self.sorting_pre_mult = [0]*len(sorting)
        for ii_dim, sp in enumerate(self.sorting):
            for s in self.sorting[ii_dim+1:]:
                if s < sp and self.X_Ai[s] == self.X_Ai[sp]:
                    self.sorting_pre_mult[ii_dim] += 1

    def __call__(self, Y_res, YA=None, *args, **kwargs):
        ii = [None]*len(self.sorting)
        if YA is None:
            YA = Y_res.conj().reshape(self.Y_shape+[-1])
            for a in self.A:
                if a is not None:
                    YA = np.tensordot(YA, a, axes=(0, 0))
                else:
                    YA = YA.transpose([ii for ii in range(1, len(YA.shape))]+[0])
        if self.normallized:
            AX = self.A.copy()
            YAX = YA.copy()
            for ii_dim, sp, spm in zip(self.sorting, self.sorting_pre, self.sorting_pre_mult):
                AX[self.X_Ai[ii_dim]] = np.tensordot(AX[self.X_Ai[ii_dim]], self.X[ii_dim], axes=(1+spm, 0))
                YAX = np.tensordot(YAX, self.X[ii_dim], axes=(1+sp, 0))
                AX_norm2 = vnorm2(AX[self.X_Ai[ii_dim]].reshape([-1, AX[self.X_Ai[ii_dim]].shape[-1]]))
                YAX_norm2 = vnorm2(YAX.reshape([-1, YAX.shape[-1]]))
                iii = np.argmax(YAX_norm2/AX_norm2)
                ii[ii_dim] = iii
                YAX = YAX[..., iii]
                AX[self.X_Ai[ii_dim]] = AX[self.X_Ai[ii_dim]][..., iii]
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
            ii = lr2hr_S(ii, Y_res, self.A, self.X_hr, self.X, self.sorting, self.normallized, YA)
        return ii

# SMOMP projection step
class SMOMP_proj:
    def __init__(self, A, X, n=2, initial=None, sorting=None, normallized=True):
        self.A = A
        self.X = X
        self.X_Ai = [None]*len(X)
        self.Y_shape = []
        ii_dim = -1
        for ii_a, a, in enumerate(A):
            if a is None:
                ii_dim += 1
                self.X_Ai[ii_dim] = ii_a
                self.Y_shape.append(self.X[ii_dim].shape[0])
            else:
                for _ in range(len(a.shape)-1):
                    ii_dim += 1
                    self.X_Ai[ii_dim] = ii_a
                self.Y_shape.append(a.shape[0])
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
            # Compute AX_ii
            X_ii = [
                compute_X_ii([
                    self.X[ii_dim][:, ii[ii_dim]]
                    for ii_dim in range(len(self.X))
                    if self.X_Ai[ii_dim] == ii_a])
                for ii_a, a in enumerate(self.A)]
            AX_ii = [
                np.tensordot(a, x_ii, axes=(
                    [ii_dim for ii_dim in range(1, len(a.shape))],
                    [ii_dim for ii_dim in range(len(x_ii.shape))]))
                if a is not None else x_ii
                for a, x_ii in zip(self.A, X_ii)]
        else:
            YA = Y_res.conj().reshape(self.Y_shape+[-1])
            for a in self.A:
                if a is not None:
                    YA = np.tensordot(YA, a, axes=(0, 0))
                else:
                    YA = YA.transpose([ii for ii in range(1, len(YA.shape))]+[0])
            ii = self.initial(Y_res, YA=YA)
        # Refinement iterations
        for _ in range(self.n):
            for ii_dim in self.sorting:
                iii = ii[ii_dim]
                # Compute projection
                if self.normallized:
                    # Exclude (ii_dim, iii) from X_ii[self.X_Ai[ii_dim]]
                    x_niii = compute_X_ii([
                        self.X[ii_dimp][:, iii]
                        for ii_dimp, iii in enumerate(ii)
                        if ii_dimp != ii_dim
                        and self.X_Ai[ii_dimp] == self.X_Ai[ii_dim]])
                    A_Xi = [
                        ii_dimp for ii_dimp, Ai in enumerate(self.X_Ai)
                        if Ai == self.X_Ai[ii_dim]]
                    td_axis_l = [
                        ii_dimp_ii+1 for ii_dimp_ii, ii_dimp in enumerate(A_Xi)
                        if ii_dimp != ii_dim]
                    td_axis_r = [ii_dimp for ii_dimp in range(len(td_axis_l))]
                    if self.A[self.X_Ai[ii_dim]] is not None:
                        ax_niii = np.tensordot(
                            self.A[self.X_Ai[ii_dim]], x_niii,
                            axes=(td_axis_l, td_axis_r))
                        ax = np.dot(ax_niii, self.X[ii_dim])
                    else:
                        ax = self.X[ii_dim]
                    # HERE!!!
                    YAX_nXiidim = np.tensordot(
                        Y_res.conj().reshape(self.Y_shape+[-1]),
                        compute_X_ii([
                            ax for ii_a, ax in enumerate(AX_ii)
                            if ii_a != self.X_Ai[ii_dim]]),
                        axes=(
                            [
                                ii_a for ii_a in range(len(self.A))
                                if ii_a != self.X_Ai[ii_dim]],
                            [ii_a for ii_a in range(len(self.A)-1)]))
                    YAX = np.tensordot(YAX_nXiidim, ax, axes=(0, 0))
                    AX_norm = vnorm(ax)
                    YAX_norm = vnorm(YAX)
                    iii = np.argmax(YAX_norm/AX_norm)
                else:
                    # Exclude (ii_dim, iii) from X_ii
                    X_niii = compute_X_ii([
                        self.X[ii_dimp][:, iii]
                        for ii_dimp, iii in enumerate(ii)
                        if ii_dimp != ii_dim])
                    td_axis_l = [
                        ii_dimp+1
                        for ii_dimp in range(len(self.X))
                        if ii_dimp != ii_dim]
                    td_axis_r = [ii_dimp for ii_dimp in range(len(self.X)-1)]
                    YAX_niii = np.tensordot(
                        YA, X_niii,
                        axes=(td_axis_l, td_axis_r))
                    YAX = np.dot(YAX_niii, self.X[ii_dim])
                    YAX_norm = vnorm(YAX)
                    iii = np.argmax(YAX_norm)
                ii[ii_dim] = iii
        # Compute X_ii
        X_ii = [
            compute_X_ii([
                self.X[ii_dim][:, ii[ii_dim]]
                for ii_dim in range(len(self.X))
                if self.X_Ai[ii_dim] == ii_a])
            for ii_a, a in enumerate(self.A)]
        AX_ii = [
            np.tensordot(a, x_ii, axes=(
                [ii_dim for ii_dim in range(1, len(a.shape))],
                [ii_dim for ii_dim in range(len(x_ii.shape))]))
            if a is not None else x_ii
            for a, x_ii in zip(self.A, X_ii)]
        AX_ii = [
            ax_ii.reshape([
                len(ax_ii) if ii_dimp == ii_dim else 1
                for ii_dimp in range(len(AX_ii))])
            for ii_dim, ax_ii in enumerate(AX_ii)]
        AX_ii_prod = 1
        for ax_ii in AX_ii:
            AX_ii_prod = AX_ii_prod * ax_ii
        AX_ii = AX_ii_prod.reshape([-1])
        return ii, AX_ii
