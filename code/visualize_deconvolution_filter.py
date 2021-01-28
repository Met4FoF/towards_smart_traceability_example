import json

import matplotlib.pyplot as plt
import numpy as np
from PyDynamic.model_estimation import invLSIIR_unc
from scipy import signal


# read (continuous) IIR transfer behavior from file
f = open("200623_MPU_9250_SN_12_X_Achse_3_COLAREF_TF_IIR.json", "r")
tf = json.load(f)
f.close()

# define some shortcuts for later use
b_sensor = tf["NumeratorCoefficient"]
ub_sensor = tf["NumeratorCoefficientUncer"]
a_sensor = tf["DenominatorCoefficient"]
ua_sensor = tf["DenominatorCoefficientUncer"]
cr_sensor = tf["CalibratedRange"]
cru_sensor = tf["CalibratedRangeUnit"]

# generate frequency response with uncertainty via Monte Carlo
mc_runs = 200
F = np.linspace(cr_sensor[0], cr_sensor[1], num=20)

AA = np.random.multivariate_normal(
    a_sensor, np.diag(np.square(ua_sensor)), size=mc_runs
)
BB = np.random.multivariate_normal(
    b_sensor, np.diag(np.square(ub_sensor)), size=mc_runs
)
HH = np.empty((mc_runs, len(F)), dtype=np.complex)

for aa, bb, i in zip(AA, BB, range(mc_runs)):
    _, HH[i, :] = signal.freqs(*signal.normalize(bb, aa), worN=F)

H_sensor = np.mean(HH, axis=0)
UH_sensor = np.cov(np.hstack((np.real(HH), np.imag(HH))).T)

# calculate amplitude and phase of sensor transfer behavior
A_sensor = np.abs(H_sensor)
P_sensor = np.unwrap(np.angle(H_sensor))

# best polynom order known from previous testing
Nb = 3
Na = 2

# assume sample rate of 1000Hz
dt = 0.001
fs = 1.0 / dt

# estimate (stabilized) inverse to compensate sensor behavior
b_inv, a_inv, _, Uab_inv = invLSIIR_unc(H_sensor, UH_sensor, Nb, Na, F, Fs=fs)

# calculate amplitude and phase of inverse filter
w_inv, H_inv = signal.freqz(b_inv, a_inv, worN=F, fs=1 / dt)
A_inv = np.abs(H_inv)
P_inv = np.unwrap(np.angle(H_inv))

# visualize
fig, ax = plt.subplots(1, 2, figsize=(10, 8), dpi=200)
color = "tab:red"
ax[0].set_xlabel("Excitation frequency $\omega$ in Hz")
ax[0].set_ylabel(r"Magnitude response $|H(\omega)|$ ", color=color)
ax[0].plot(F, A_inv, markersize=10, label="deconvolution filter", color=color,)
ax[0].plot(F, A_sensor * A_inv, "o", label="compensation effect", color=color, lw=2)
ax[0].tick_params(axis="y", labelcolor=color)
ax[0].legend()

color = "tab:blue"
ax[1].set_ylabel(
    r"Phase respone $\Delta \phi$ in °", color=color
)  # we already handled the x-label with ax1
ax[1].set_xlabel("Excitation frequency $\omega$ in Hz")
ax[1].plot(
    F, P_inv / np.pi * 180, color=color, markersize=10, label="deconvolution filter"
)
ax[1].plot(F, (P_sensor + P_inv) / np.pi * 180, "o", label="compensation effect", color=color, lw=2)
ax[1].tick_params(axis="y", labelcolor=color)
ax[1].legend()
plt.show()
