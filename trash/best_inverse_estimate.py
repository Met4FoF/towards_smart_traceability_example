import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# load goal
f = open(
    "data\\200623_MPU_9250_SN_12_X_Achse_3_COLAREF\\200623_MPU_9250_SN_12_X_Achse_3_COLAREF_TF_IIR.json",
    "r",
)
tf = json.load(f)
f.close()

b_goal = tf["NumeratorCoefficient"]
a_goal = tf["DenominatorCoefficient"]
cr = tf["CalibratedRange"]

F = np.linspace(cr[0], cr[1], num=50)
w_goal, H_goal = signal.freqs(b_goal, a_goal, worN=F)
A_goal = np.abs(H_goal)
P_goal = np.unwrap(np.angle(H_goal))

# load fitted inverse filters
f = open("fit_result_tau2.json", "r")
res = json.load(f)
f.close()

dt = 0.001

# plot fitted frequency response
fig, ax = plt.subplots(nrows=2)
ax[0].plot(F, A_goal, ":", label=f"goal")
ax[1].plot(F, P_goal, ":", label=f"goal")

for Na, val in res.items():
    for Nb, filter_dict in val.items():
        a = filter_dict["a"]
        b = filter_dict["b"]

        w, H = signal.freqz(b, a, worN=F, fs=1/dt)

        A = np.abs(H)
        P = np.unwrap(np.angle(H))

        delta_amplitude = np.linalg.norm(1 - A * A_goal)
        delta_phase = np.linalg.norm(P + P_goal)

        if delta_phase < 15 and delta_amplitude < 0.5:
            ax[0].plot(F, A, label=f"{Na}, {Nb}")
            ax[1].plot(F, P, label=f"{Na}, {Nb}")

            print(f"Na{Na}, Nb{Nb}: delta_amplitude = {delta_amplitude} | delta_phase = {delta_phase}")


ax[0].legend()
plt.show()