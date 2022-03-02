import numpy as np

def OMP(Y, A, n):
    bool_flat = len(Y.shape) == 1
    if bool_flat:
        Y = Y[:, np.newaxis]
    X_norm = normallized(X)
    Y_res = Y.copy()
    I = []
    for _ in range(n):
        YX = np.dot(np.conj(Y_res.T), X_norm)
        I.append(np.argmax(np.linalg.norm(YX, axis = 0)))
        alpha = np.linalg.lstsq(X[:, I], Y, rcond = None)[0]
        Y_res = Y - np.dot(X[:, I], alpha)
    if bool_flat:
        alpha = alpha[:, 0]
    return alpha, I

def MOMP(Y, A, X, n, sorting = None, refinement = 5, AX_all_norm = None):
    if sorting is None:
        sorting = np.argsort([x.shape[0] for x in X])[::-1]         # Bigger dimensions are computed first
    N_dims = len(sorting)
    bool_flat = len(Y.shape) == 1
    if bool_flat:
        Y = Y[..., np.newaxis]
    N_measures = Y.shape[-1]
    Size_measures = A.shape[-1]
    # Inverse permutation of sorting
    sorting_inv = np.arange(N_dims)
    sorting_inv[sorting] = np.arange(N_dims)
    # Apply sorting
    A = np.transpose(A, [N_dims]+[s for s in sorting])
    if AX_all_norm is not None:
        if AX_all_norm != "ignore":
            AX_all_norm = np.transpose(AX_all_norm, sorting)
    # Initialization
    I = []
    Y_res = Y
    # Loop
    while True:
        pass

    # Undo flatness
    if bool_flat:
        alpha = alpha[:, 0]
    return alpha, np.array(I)[:, sorting_inv]
