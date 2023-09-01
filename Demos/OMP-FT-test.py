import MOMP
import numpy as np
from time import time

# Params
ITER = 100                      # Number of Monte-Carlo simulations
N_s = 35                        # Dictionary atoms size
N_a = 1024                      # Dictionary atoms
N_m = 2                         # Number of observations
N_p = 1                         # Number of features

# Create dictionaries
domain = np.linspace(0, 2*np.pi, N_a, endpoint=False)
X = np.exp(1j*np.arange(N_s)[:, np.newaxis]*domain[np.newaxis])

# Define stop criteria
stop = MOMP.stop.General(maxIter=N_p)   # We assume we know the number of paths

# Define projection step
proj = MOMP.proj.OMP_proj(X)
proj_FT = MOMP.proj.OMP_proj_FT()

# Define algorithm
alg_vanilla = MOMP.mp.OMP(proj, stop)
alg_FT = MOMP.mp.OMP(proj_FT, stop)
algs = [alg_vanilla, alg_FT]
alg_names = ["Vanilla", "FT"]

# Define evaluation metric
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
def metric(feat, feat_est):
    return np.mean([
        np.min(np.linalg.norm(wrap_angle(feat_est - ff[np.newaxis]), ord=2))
        for ff in feat])

# Initialization
CTime = np.zeros((len(algs), ITER))
Metric = np.zeros((len(algs), ITER))

# Iterate over different channels
for iter in range(ITER):
    print("{}/{}".format(iter, ITER))
    # Generate features
    feat = np.random.uniform(0, 2*np.pi, size=N_p)
    alpha = np.random.randn(N_p, N_m) + 1j*np.random.randn(N_p, N_m)

    # Compute sparse signal
    H = np.zeros([N_s, N_m], dtype="complex128")
    for ii_path in range(N_p):
        H_path = np.exp(1j*np.arange(N_s)[:]*feat[ii_path])
        shape = [1]*len(H.shape)
        shape[-1] = N_m
        H += H_path[..., np.newaxis]*alpha[ii_path]

    # Compute observation
    Y = H.reshape([-1, N_m])

    # Try different algorithms
    for ii_alg, alg in enumerate(algs):
        tic = time()
        I, alpha = alg(Y)
        CTime[ii_alg, iter] = time()-tic
        # Retrieve features
        try:
            feat_est = domain[I]
        except:
            feat_est = I
        # Evaluate
        Metric[ii_alg, iter] = metric(feat, feat_est)

# Print
for met_med, alg_name, ctime in zip(np.median(Metric, axis=1), alg_names, np.mean(CTime, axis=1)):
    print("Algorithm {} achieves a median metric of {:.3f} in {:.2}s".format(alg_name, met_med, ctime))

# Plots
import matplotlib.pyplot as plt

plt.figure("Metric")
for ii_alg, alg_name in enumerate(alg_names):
    plt.plot(np.sort(Metric[ii_alg]), np.linspace(0, 100, ITER), label = alg_name)
plt.xlim([0, 0.1])
plt.xlabel("Metric")
plt.ylabel("Probability [%]")
plt.legend()

plt.figure("CTime")
plt.bar(alg_names, np.mean(CTime, axis=1))
plt.yscale("log")
plt.ylabel("Computational time [s]")
plt.show()
