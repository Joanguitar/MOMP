import MOMP
import numpy as np
from time import time
from line_profiler import LineProfiler

# Params
ITER = 100                      # Number of Monte-Carlo simulations
N_s = [4]*3+[5, 6, 3]           # Dictionary atoms size
N_a = [80]*5+[100]              # Dictionary atoms
N_a_lr = [16]*5+[20]            # Dictionary atoms (low resolution)
D1 = 3                          # Number of dictionaries associated to the first measurement dimention
N_o1 = 50                       # Observation length
N_o2 = 20                       # Observation length
N_m = 2                         # Number of observations
N_p = 1                         # Number of features

# Extra params
N_o = N_o1*N_o2

# Create dictionaries
Domains = [np.linspace(0, 2*np.pi, na, endpoint=False) for na in N_a]
X = [np.exp(1j*np.arange(ns)[:, np.newaxis]*domain[np.newaxis]) for ns, domain in zip(N_s, Domains)]

# Create low-resolution dictionaries
Domains_lr = [np.linspace(0, 2*np.pi, na, endpoint=False) for na in N_a_lr]
X_lr = [np.exp(1j*np.arange(ns)[:, np.newaxis]*domain[np.newaxis]) for ns, domain in zip(N_s, Domains_lr)]

# Create random A
A1 = np.random.randn(*([N_o1]+[N_s[ii] for ii in range(D1)])) + 1j*np.random.randn(*([N_o1]+[N_s[ii] for ii in range(D1)]))
A2 = np.random.randn(*([N_o2]+[N_s[ii] for ii in range(D1, len(N_s))])) + 1j*np.random.randn(*([N_o2]+[N_s[ii] for ii in range(D1, len(N_s))]))
A = A1.reshape([N_o1, 1]+[N_s[ii] for ii in range(D1)]+[1 for ii in range(D1, len(N_s))]) * A2.reshape([1, N_o2]+[1 for ii in range(D1)]+[N_s[ii] for ii in range(D1, len(N_s))])
A = A.reshape([N_o]+N_s)

# Define stop criteria
stop = MOMP.stop.General(maxIter=N_p)   # We assume we know the number of paths

# Define initial projection step
proj_init_MOMP = MOMP.proj.MOMP_greedy_proj(A, X, X_lr, normallized=False)
proj_init_SMOMP = MOMP.proj.SMOMP_greedy_proj([A1, A2], X, X_lr, normallized=False)

# Define projection step
proj_MOMP_nonorm = MOMP.proj.MOMP_proj(A, X, initial=proj_init_MOMP, normallized=False)
proj_MOMP = MOMP.proj.MOMP_proj(A, X, initial=proj_init_MOMP, normallized=True)
proj_SMOMP_nonorm = MOMP.proj.SMOMP_proj([A1, A2], X, initial=proj_init_SMOMP, normallized=False)
proj_SMOMP = MOMP.proj.SMOMP_proj([A1, A2], X, initial=proj_init_SMOMP, normallized=True)

# Define algorithm
alg_MOMP_nonorm = MOMP.mp.OMP(proj_MOMP_nonorm, stop)
alg_MOMP = MOMP.mp.OMP(proj_MOMP, stop)
alg_SMOMP_nonorm = MOMP.mp.OMP(proj_SMOMP_nonorm, stop)
alg_SMOMP = MOMP.mp.OMP(proj_SMOMP, stop)
algs = [alg_SMOMP, alg_MOMP, alg_SMOMP_nonorm, alg_MOMP_nonorm]
alg_names = ["SMOMP", "MOMP", "SMOMP-nonorm", "MOMP-nonorm"]

# Define evaluation metric
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
def metric(feat, feat_est):
    return np.mean([
        np.min(np.linalg.norm(wrap_angle(feat_est - ff[np.newaxis]), ord=2, axis=1))
        for ff in feat])

# Initialization
CTime = np.zeros((len(algs), ITER))
Metric = np.zeros((len(algs), ITER))

# Iterate over different channels
for iter in range(ITER):
    print("{}/{}".format(iter, ITER))
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

    # Try different algorithms
    for ii_alg, alg in enumerate(algs):
        tic = time()
        I, alpha = alg(Y)
        CTime[ii_alg, iter] = time()-tic
        # Retrieve features
        feat_est = np.asarray([[dom[iii] for iii, dom in zip(ii, Domains)] for ii in I])
        # Evaluate
        Metric[ii_alg, iter] = metric(feat, feat_est)

# Print
for met_med, alg_name, ctime in zip(np.median(Metric, axis=1), alg_names, np.mean(CTime, axis=1)):
    print("Algorithm {} achieves a median metric of {:.3f} in {:.2}s".format(alg_name, met_med, ctime))

# Plots
import matplotlib.pyplot as plt

Style = {"MOMP": "--", "SMOMP": "-"}

plt.figure("Metric")
for ii_alg, alg_name in enumerate(alg_names):
    plt.plot(np.sort(Metric[ii_alg]), np.linspace(0, 100, ITER), Style[alg_name.split("-")[0]], label = alg_name)
plt.xlim([0, 0.5])
plt.xlabel("Metric")
plt.ylabel("Probability [%]")
plt.legend()

plt.figure("CTime")
plt.bar(alg_names, np.mean(CTime, axis=1))
plt.yscale("log")
plt.ylabel("Computational time [s]")
plt.show()
