import numpy as np
import h5py
import json
import matplotlib.pyplot as plt


file_raw = "dump/ConnectorAgent_1.h5"
file_inp = "dump/InterpolationAgent_1.h5"
file_dcv = "dump/DeconvolutionAgent_1.h5"

raw = h5py.File(file_raw, "r")
inp = h5py.File(file_inp, "r")
dcv = h5py.File(file_dcv, "r")

i = 3
start = 330000
stop = 340000
indices = np.arange(start,stop)
t_raw = raw["time_series/t"][indices]
v_raw = raw["time_series/v"][indices, i]
uv_raw = raw["time_series/uv"][indices, i]
t_inp = inp["time_series/t"][indices]
v_inp = inp["time_series/v"][indices, i]
uv_inp = inp["time_series/uv"][indices, i]
t_dcv = dcv["time_series/t"][indices]
v_dcv = dcv["time_series/v"][indices, i]
uv_dcv = dcv["time_series/uv"][indices, i]


# labels from metadata
meta = raw["time_series"].attrs
t_name = json.loads(meta["time_name"])
t_unit = json.loads(meta["time_unit"])
q_names = json.loads(meta["quantity_names"])
q_units = json.loads(meta["quantity_units"])
xlabel = f"{t_name} / [{t_unit}]"
ylabel = f"{q_names[i]} / [{q_units[i]}]"

# visualize
plt.plot(t_raw, v_raw, "o", markersize=1, label="raw")
plt.plot(t_dcv, v_dcv, label="deconvolved")
plt.fill_between(t_dcv, v_dcv - uv_dcv, v_dcv + uv_dcv, color="orange", alpha=0.2)
#plt.plot(t_inp, v_inp, label="interpolated")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()
plt.show()

raw.close()
dcv.close()