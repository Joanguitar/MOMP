# This script also requires my package pywarraychannels which you can install like
# pip install git+https://github.com/Joanguitar/pywarraychannels

import pywarraychannels
import numpy as np
import scipy
import MOMP
from pathlib import Path

# Params
link = "up"             # Whether it's up-link or down-link
method = "SMOMP"        # Channel estimation method (MOMP or OMP)
K_res = 512             # Method's dictionary resolution
K_res_lr = 4            # Method's dictionary low resolution
samples = None          # Number of samples from the dataset to evaluate
# Power
p_t_dBm = 20            # dBm
# Noise related
T = 15                  # C
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 3e8                 # m/s
# Antennas
N_UE = 4                # Number of UE antennas in each dimension
N_AP = 8                # Number of AP antennas in each dimension
N_RF_UE = N_UE          # Number of UE RF-chains in total
N_RF_AP = N_AP          # Number of AP RF-chains in total
N_M_UE = N_UE           # Number of UE measurements in each dimension
N_M_AP = N_AP           # Number of AP measurements in each dimension
# Carriers
f_c = 60                # Centra frequency [GHz]
B = 2                   # Bandwidth [GHz]
K = 64                  # Number of delay taps
Q = 64                  # Length of the training pilot
# Estimation
N_est = 5               # Number of estimated paths

# Define pulse shape filter
filter = pywarraychannels.filters.RCFilter(early_samples=8, late_samples=8)

# Pilot signals
if link == "up":
    Pilot = np.concatenate([scipy.linalg.hadamard(Q)[:N_RF_UE], np.zeros((N_RF_UE, K//2))], axis=1)
else:
    Pilot = np.concatenate([scipy.linalg.hadamard(Q)[:N_RF_AP], np.zeros((N_RF_AP, K//2))], axis=1)
P_len = Pilot.shape[1]
D = K+filter.early_samples+filter.late_samples
Pilot = np.concatenate([np.zeros((Pilot.shape[0], D)), Pilot], axis=1)      # Zero-padding

# Define antennas as rectangular with same number of antennas in each dimension
antenna_UE = pywarraychannels.antennas.RectangularAntenna((N_UE, N_UE), z_positive=True)
antenna_AP = pywarraychannels.antennas.RectangularAntenna((N_AP, N_AP), z_positive=True)
# Define the orientation of the AP as mounted on the wall
antenna_AP.uncertainty = pywarraychannels.uncertainties.Static(tilt=np.pi/2)

# Define codebooks (This function applies the baseline beam-pattern design of my paper "Lightweight and effective sector beam pattern synthesis with uniform linear antenna arrays" for the corresponding beam-width)
antenna_UE.set_reduced_codebook((N_M_UE, N_M_UE))
antenna_AP.set_reduced_codebook((N_M_AP, N_M_AP))

# Split codebooks according to number of RF-chains so measurements can be taken by groups of beam-patterns according to the number of RF-chains
cdbks_UE = np.transpose(np.reshape(antenna_UE.codebook, [N_UE**2, -1, N_RF_UE]), [1, 0, 2])
cdbks_AP = np.transpose(np.reshape(antenna_AP.codebook, [N_AP**2, -1, N_RF_AP]), [1, 0, 2])

# Transform params to natural units
f_c *= 1e9
B *= 1e9
T += 273.1
p_t = np.power(10, (p_t_dBm-30)/10)

# Compute noise level
p_n = k_B*T*B

# Define channel as Geomtric-MIMO-AWGN
if link == "up":
    channel_Geometric = pywarraychannels.channels.Geometric(
        antenna_AP, antenna_UE, f_c=f_c,
        B=B, K=K, filter=filter, bool_sync=True)
else:
    channel_Geometric = pywarraychannels.channels.Geometric(
        antenna_UE, antenna_AP, f_c=f_c,
        B=B, K=K, filter=filter, bool_sync=True)
channel_MIMO = pywarraychannels.channels.MIMO(channel_Geometric, pilot=Pilot)
channel = pywarraychannels.channels.AWGN(channel_MIMO, power=p_t, noise=p_n)

# Whitening matrices that transform each combiner's noise distribution to white
if link == "up":
    LLinv = [np.linalg.inv(np.linalg.cholesky(np.dot(np.conj(cdbk.T), cdbk))) for cdbk in cdbks_AP]
else:
    LLinv = [np.linalg.inv(np.linalg.cholesky(np.dot(np.conj(cdbk.T), cdbk))) for cdbk in cdbks_UE]

# Measurement matrices related to SMOMP
if link == "up":
    L_invW = []
    for cdbk_AP, Linv in zip(cdbks_AP, LLinv):
        L_invW.append(np.dot(Linv, np.conj(cdbk_AP.T)))
    L_invW = np.concatenate(L_invW, axis=0).reshape([-1, N_AP, N_AP])       # N_M_RX  x  N_RX  x  N_RX
    FE_conv = []
    for cdbk_UE in cdbks_UE:
        FE = np.dot(cdbk_UE, Pilot)
        FE_conv.append(np.zeros((N_UE**2, D, P_len), dtype="complex128"))
        for k in range(D):
            FE_conv[-1][:, k, :] = FE[:, D-k:P_len+D-k]
    FE_conv = np.concatenate(FE_conv, axis=2)
    FE_conv = FE_conv.transpose([2, 0, 1]).reshape([-1, N_UE, N_UE, D])     # (P_len*N_M_TX/N_RF_TX) x N_TX x N_TX x D

    L_invW_U, _, _ = np.linalg.svd(L_invW.reshape([-1, N_AP*N_AP]), full_matrices=False)
    FE_conv_U, _, _, = np.linalg.svd(FE_conv.reshape([-1, N_UE*N_UE*D]), full_matrices=False)

    L_invW_x_U = np.tensordot(L_invW_U.conj(), L_invW, axes=(0, 0))
    FE_conv_x_U = np.tensordot(FE_conv_U.conj(), FE_conv, axes=(0, 0))
else:
    L_invW = []
    for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
        L_invW.append(np.dot(Linv, np.conj(cdbk_UE.T)))
    L_invW = np.concatenate(L_invW, axis=0).reshape([-1, N_UE, N_UE])       # N_M_RX  x  N_RX  x  N_RX
    FE_conv = []
    for cdbk_AP in cdbks_AP:
        FE = np.dot(cdbk_AP, Pilot)
        FE_conv.append(np.zeros((N_AP**2, D, P_len), dtype="complex128"))
        for k in range(D):
            FE_conv[-1][:, k, :] = FE[:, D-k:P_len+D-k]
    FE_conv = np.concatenate(FE_conv, axis=2)
    FE_conv = FE_conv.transpose([2, 0, 1]).reshape([-1, N_AP, N_AP, D])     # (P_len*N_M_TX/N_RF_TX) x N_TX x N_TX x D

    L_invW_U, _, _ = np.linalg.svd(L_invW.reshape([-1, N_UE*N_UE]), full_matrices=False)
    FE_conv_U, _, _, = np.linalg.svd(FE_conv.reshape([-1, N_AP*N_AP*D]), full_matrices=False)

    L_invW_x_U = np.tensordot(L_invW_U.conj(), L_invW, axes=(0, 0))
    FE_conv_x_U = np.tensordot(FE_conv_U.conj(), FE_conv, axes=(0, 0))
# Merge the multiple measure matrices into a single tensor for MOMP
if method == "MOMP":
    if link == "up":
        A = L_invW_x_U.reshape((-1, 1, N_AP, N_AP, 1, 1, 1)) * FE_conv_x_U.reshape((1, -1, 1, 1, N_UE, N_UE, D))
        A = A.reshape((-1, N_AP, N_AP, N_UE, N_UE, D))
    else:
        A = L_invW_x_U.reshape((-1, 1, N_UE, N_UE, 1, 1, 1)) * FE_conv_x_U.reshape((1, -1, 1, 1, N_AP, N_AP, D))
        A = A.reshape((-1, N_UE, N_UE, N_AP, N_AP, D))

# Sparse decomposition components
angles_AP = np.linspace(-np.pi, np.pi, int(N_M_AP*K_res))
angles_UE = np.linspace(-np.pi, np.pi, int(N_M_UE*K_res))
A_AP = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP[np.newaxis, :])
A_UE = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE[np.newaxis, :])
delays = np.linspace(0, K, int(K*K_res))
A_time = filter.response(K, delays)

# Sparse decomposition components
angles_AP_lr = np.linspace(-np.pi, np.pi, int(N_M_AP*K_res_lr))
angles_UE_lr = np.linspace(-np.pi, np.pi, int(N_M_UE*K_res_lr))
A_AP_lr = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP_lr[np.newaxis, :])
A_UE_lr = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE_lr[np.newaxis, :])
delays_lr = np.linspace(0, K, int(K*K_res_lr))
A_time_lr = filter.response(K, delays_lr)

# Dictionaries
if link == "up":
    X = [
        A_AP,
        A_AP,
        np.conj(A_UE),
        np.conj(A_UE),
        A_time
    ]
    X_lr = [
        A_AP_lr,
        A_AP_lr,
        np.conj(A_UE_lr),
        np.conj(A_UE_lr),
        A_time_lr
    ]
else:
    X = [
        A_UE,
        A_UE,
        np.conj(A_AP),
        np.conj(A_AP),
        A_time
    ]
    X_lr = [
        A_UE_lr,
        A_UE_lr,
        np.conj(A_AP_lr),
        np.conj(A_AP_lr),
        A_time_lr
    ]

# Define decomposition algorithm
stop = MOMP.stop.General(maxIter=N_est)     # Stop when reached the desired number of estimated paths
if method == "MOMP":
    proj_init = MOMP.proj.MOMP_greedy_proj(A, X, X_lr, normallized=False)
    proj = MOMP.proj.MOMP_proj(A, X, initial=proj_init, normallized=False)
elif method == "SMOMP":
    proj_init = MOMP.proj.SMOMP_greedy_proj((L_invW_x_U, FE_conv_x_U), X, X_lr, normallized=False)
    proj = MOMP.proj.SMOMP_proj((L_invW_x_U, FE_conv_x_U), X, initial=proj_init, normallized=False)
else:
    print("Method {} not implemented".format(method))
    raise
alg = MOMP.mp.OMP(proj, stop)

# Load data
with open(Path(__file__).with_name("rays.txt")) as f:
    rays = pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in f.read().split("\n")])

# Build channel
channel.build(rays)
# Simulate measurement
MM = []
if link == "up":
    for cdbk_AP, Linv in zip(cdbks_AP, LLinv):
        MMM = []
        antenna_AP.set_codebook(cdbk_AP)
        channel.set_corr(np.dot(np.conj(antenna_AP.codebook.T), antenna_AP.codebook))
        for cdbk_UE in cdbks_UE:
            antenna_UE.set_codebook(cdbk_UE)
            MMM.append(np.dot(Linv, channel.measure()))
        MM.append(MMM)
else:
    for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
        MMM = []
        antenna_UE.set_codebook(cdbk_UE)
        channel.set_corr(np.conj(antenna_UE.codebook.T), antenna_UE.codebook)
        for cdbk_AP in cdbks_AP:
            antenna_AP.set_codebook(cdbk_AP)
            MM.append(np.dot(Linv, channel.measure()))
        MM.append(MMM)
# Compile measurements into a matrix
M = np.concatenate([np.concatenate(MMM, axis=1) for MMM in MM], axis=0)
M_U = np.tensordot(np.tensordot(M, L_invW_U.conj(), axes=(0, 0)), FE_conv_U.conj(), axes=(0, 0))
# Apply sparse recovery algorithm
I, alpha = alg(M_U.reshape([-1]))
if method == "OMP":
    I = [list(np.unravel_index(ii, [x.shape[1] for x in X])) for ii in I]
# Recover the channel parameters
Alpha = []
Power = []
DoA = []
DoD = []
ToF = []
for a, iii in zip(alpha, I):
    Alpha.append(a)
    Power.append(20*np.log10(np.linalg.norm(a)))
    ii_component = 0
    if link == "up":
        xoa, yoa = [angles_AP[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
    else:
        xoa, yoa = [angles_UE[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
    zoa = xoa**2 + yoa**2
    if zoa > 1:
        xoa, yoa = xoa/np.sqrt(zoa), yoa/np.sqrt(zoa)
        zoa = 0
    else:
        zoa = np.sqrt(1-zoa)
    doa = np.array([xoa, yoa, zoa])
    DoA.append(doa)
    ii_component += 2
    if link == "up":
        xod, yod = [angles_UE[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
    else:
        xod, yod = [angles_AP[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
    zod = xod**2 + yod**2
    if zod > 1:
        xod, yod = xod/np.sqrt(zod), yod/np.sqrt(zod)
        zod = 0
    else:
        zod = np.sqrt(1-zod)
    dod = np.array([xod, yod, zod])
    DoD.append(dod)
    ii_component += 2
    tof = delays[iii[ii_component]]
    ToF.append(tof)
Alpha = np.array(Alpha)
Power = np.array(Power)
DoA = np.array(DoA)
DoD = np.array(DoD)
if link == "up":
    DoA = antenna_AP.uncertainty.apply(DoA)
    DoD = antenna_UE.uncertainty.apply(DoD)
else:
    DoA = antenna_UE.uncertainty.apply(DoA)
    DoD = antenna_AP.uncertainty.apply(DoD)
ToF = np.array(ToF)/B
TDoF = ToF - ToF[0]
DDoF = TDoF*c
DoA_az, DoA_el = pywarraychannels.em.cartesian2polar(DoA)
DoD_az, DoD_el = pywarraychannels.em.cartesian2polar(DoD)
# Compose estimation ray information
rays_est = pywarraychannels.em.Geometric(np.array([
    np.rad2deg(np.angle(Alpha)),
    TDoF,
    10*np.log10(np.abs(Alpha)**2)-(p_t_dBm-30)+30,
    np.rad2deg(DoA_az),
    np.rad2deg(DoA_el),
    np.rad2deg(DoD_az),
    np.rad2deg(DoD_el)
]).T)

# Display information (Note that phase and time information are impossible to compute without accurate knowledge of when the transmission was triggered, thus the estimation of these is relative)
print("Ground truth:")
print(rays.first(5))
print("Estimation:")
print(rays_est.first(5))