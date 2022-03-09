import MOMP
import numpy as np
from time import time
from line_profiler import LineProfiler

# Params
ITER = 100                      # Number of Monte-Carlo simulations
N_s1 = [4]*3+[5, 6, 3]          # Dictionary atoms size
N_a1 = [80]*5+[100]             # Dictionary atoms
N_a1_lr = [16]*5+[20]           # Dictionary atoms (low resolution)
N_s2 = [3]*3+[4, 5, 6]          # Dictionary atoms size
N_a2 = [60]*5+[120]             # Dictionary atoms
N_a2_lr = [12]*5+[24]           # Dictionary atoms (low resolution)
N_o = 1000                      # Observation length
N_m = 2                         # Number of observations
N_p1 = 1                        # Number of features
N_p2 = 1                        # Number of features

# Create dictionaries
Domains1 = [np.linspace(0, 2*np.pi, na, endpoint=False) for na in N_a1]
X1 = [np.exp(1j*np.arange(ns)[:, np.newaxis]*domain[np.newaxis]) for ns, domain in zip(N_s1, Domains1)]
Domains2 = [np.linspace(0, 2*np.pi, na, endpoint=False) for na in N_a2]
X2 = [np.exp(1j*np.arange(ns)[:, np.newaxis]*domain[np.newaxis]) for ns, domain in zip(N_s2, Domains2)]

# Create low-resolution dictionaries
Domains1_lr = [np.linspace(0, 2*np.pi, na, endpoint=False) for na in N_a1_lr]
X1_lr = [np.exp(1j*np.arange(ns)[:, np.newaxis]*domain[np.newaxis]) for ns, domain in zip(N_s1, Domains1_lr)]
Domains2_lr = [np.linspace(0, 2*np.pi, na, endpoint=False) for na in N_a2_lr]
X2_lr = [np.exp(1j*np.arange(ns)[:, np.newaxis]*domain[np.newaxis]) for ns, domain in zip(N_s2, Domains2_lr)]

# Create random As
A1 = np.random.randn(*([N_o]+N_s1)) + 1j*np.random.randn(*([N_o]+N_s1))
A2 = np.random.randn(*([N_o]+N_s2)) + 1j*np.random.randn(*([N_o]+N_s2))

# Define stop criteria
stop = MOMP.stop.General(maxIter=N_p1+N_p2)   # We assume we know the number of paths

# Define initial projection step
proj1_init_lr = MOMP.proj.MOMP_greedy_proj(A1, X1, X1_lr)
proj1_init_lr_nonorm = MOMP.proj.MOMP_greedy_proj(A1, X1, X1_lr, normallized=False)
proj2_init_lr = MOMP.proj.MOMP_greedy_proj(A2, X2, X2_lr)
proj2_init_lr_nonorm = MOMP.proj.MOMP_greedy_proj(A2, X2, X2_lr, normallized=False)

# Define projection step
proj1 = MOMP.proj.MOMP_proj(A1, X1)
proj1_nonorm = MOMP.proj.MOMP_proj(A1, X1, normallized=False)
proj1_lr = MOMP.proj.MOMP_proj(A1, X1, initial=proj1_init_lr)
proj1_lr_nonorm = MOMP.proj.MOMP_proj(A1, X1, initial=proj1_init_lr_nonorm, normallized=False)
proj2 = MOMP.proj.MOMP_proj(A2, X2)
proj2_nonorm = MOMP.proj.MOMP_proj(A2, X2, normallized=False)
proj2_lr = MOMP.proj.MOMP_proj(A2, X2, initial=proj2_init_lr)
proj2_lr_nonorm = MOMP.proj.MOMP_proj(A2, X2, initial=proj2_init_lr_nonorm, normallized=False)

# Define algorithm
alg_vanilla = MOMP.mp.OMPmultiproj([proj1, proj2], stop)
alg_nonorm = MOMP.mp.OMPmultiproj([proj1_nonorm, proj2_nonorm], stop)
alg_lr = MOMP.mp.OMPmultiproj([proj1_lr, proj2_lr], stop)
alg_lr_nonorm = MOMP.mp.OMPmultiproj([proj1_lr_nonorm, proj2_lr_nonorm], stop)
algs = [alg_vanilla, alg_nonorm, alg_lr, alg_lr_nonorm]
alg_names = ["Vanilla", "No norm", "LR", "LR - No norm"]

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
    feat1 = np.random.uniform(0, 2*np.pi, size=[N_p1, len(N_a1)])
    alpha1 = np.random.randn(N_p1, N_m) + 1j*np.random.randn(N_p1, N_m)
    feat2 = np.random.uniform(0, 2*np.pi, size=[N_p2, len(N_a2)])
    alpha2 = np.random.randn(N_p2, N_m) + 1j*np.random.randn(N_p2, N_m)

    # Compute sparse signal
    H1 = np.zeros(N_s1+[N_m], dtype="complex128")
    for ii_path in range(N_p1):
        H_path = np.ones_like(H1[..., 0])
        for ii_dim, ns in enumerate(N_s1):
            shape = [1]*len(N_s1)
            shape[ii_dim] = ns
            H_path *= np.reshape(np.exp(1j*np.arange(ns)[:]*feat1[ii_path, ii_dim]), shape)
        shape = [1]*len(H1.shape)
        shape[-1] = N_m
        H1 += H_path[..., np.newaxis]*np.reshape(alpha1[ii_path], shape)
    H2 = np.zeros(N_s2+[N_m], dtype="complex128")
    for ii_path in range(N_p2):
        H_path = np.ones_like(H2[..., 0])
        for ii_dim, ns in enumerate(N_s2):
            shape = [1]*len(N_s2)
            shape[ii_dim] = ns
            H_path *= np.reshape(np.exp(1j*np.arange(ns)[:]*feat2[ii_path, ii_dim]), shape)
        shape = [1]*len(H2.shape)
        shape[-1] = N_m
        H2 += H_path[..., np.newaxis]*np.reshape(alpha2[ii_path], shape)

    # Compute observation
    Y = A1.reshape([N_o, -1]) @ H1.reshape([-1, N_m]) + A2.reshape([N_o, -1]) @ H2.reshape([-1, N_m])

    # Try different algorithms
    for ii_alg, alg in enumerate(algs):
        tic = time()
        I_p, I, alpha = alg(Y)
        CTime[ii_alg, iter] = time()-tic
        # Retrieve features
        feat1_est = np.asarray([[dom[iii] for iii, dom in zip(ii, Domains1)] for ii_p, ii in zip(I_p, I) if ii_p == 0])
        feat2_est = np.asarray([[dom[iii] for iii, dom in zip(ii, Domains2)] for ii_p, ii in zip(I_p, I) if ii_p == 1])
        # Evaluate
        if feat1.size > 0:
            Metric[ii_alg, iter] += metric(feat1, feat1_est)
        if feat2.size > 0:
            Metric[ii_alg, iter] += metric(feat2, feat2_est)

# Print
for met_med, alg_name, ctime in zip(np.median(Metric, axis=1), alg_names, np.mean(CTime, axis=1)):
    print("Algorithm {} achieves a median metric of {:.3f} in {:.2}s".format(alg_name, met_med, ctime))

# Plots
import matplotlib.pyplot as plt

plt.figure("Metric")
for ii_alg, alg_name in enumerate(alg_names):
    plt.plot(np.sort(Metric[ii_alg]), np.linspace(0, 100, ITER), label = alg_name)
plt.xlim([0, 0.5])
plt.xlabel("Metric")
plt.ylabel("Probability [%]")
plt.legend()

plt.figure("CTime")
plt.bar(alg_names, np.mean(CTime, axis=1))
plt.yscale("log")
plt.ylabel("Computational time [s]")
plt.show()
