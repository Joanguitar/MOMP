import MOMP
import numpy as np

# Params
N_s = [5]*5+[6]                 # Dictionary atoms size
N_a = [100]*5+[120]             # Dictionary atoms
N_o = 1000                      # Observation length
N_m = 2                         # Number of observations
N_p = 1                         # Number of features

# Create dictionaries
Domains = [np.linspace(0, 2*np.pi, na, endpoint=False) for na in N_a]
X = [np.exp(1j*np.arange(ns)[:, np.newaxis]*domain[np.newaxis]) for ns, domain in zip(N_s, Domains)]

# Create random A
A = np.random.randn(*([N_o]+N_s)) + 1j*np.random.randn(*([N_o]+N_s))

# Generate features
feat = np.random.uniform(0, 2*np.pi, size=[N_p, len(N_a)])
alpha = np.random.randn(N_p, N_m) + 1j*np.random.randn(N_p, N_m)

# Compute sparse signal
H = np.zeros(N_s+[N_m], dtype="complex128")
for ii_path in range(N_p):
    H_path = np.ones_like(H[..., 0])
    for ii_dim, ns in enumerate(N_s):
        shape = [1]*len(N_s)
        shape[ii_dim] = ns
        H_path *= np.reshape(np.exp(1j*np.arange(ns)[:]*feat[ii_path, ii_dim]), shape)
    shape = [1]*len(H.shape)
    shape[-1] = N_m
    H += H_path[..., np.newaxis]*np.reshape(alpha[ii_path], shape)

# Compute observation
Y = A.reshape([N_o, -1]) @ H.reshape([-1, N_m])

# Decompose Y
#alpha, I = MOMP(Y, A, X, 3)

# Test
stop = MOMP.stop.General(maxIter=5)
proj = MOMP.proj.MOMP_proj(A, X)
alg = MOMP.mp.OMP(proj, stop)
I, alpha = alg(Y)
proj = MOMP.proj.MOMP_proj(A, X, normallized=False)
alg = MOMP.mp.OMP(proj, stop)
I_fast, alpha = alg(Y)

# Retrieve features
feat_est = [[dom[iii] for iii, dom in zip(ii, Domains)] for ii in I]
feat_fast_est = [[dom[iii] for iii, dom in zip(ii, Domains)] for ii in I_fast]
print(feat, feat_est, feat_fast_est)
