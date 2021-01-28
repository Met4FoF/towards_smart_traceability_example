import copy
import json
import time
import os

import h5py
import numpy as np
from agentMET4FOF.metrological_agents import MetrologicalAgent, MetrologicalMonitorAgent
from time_series_metadata.scheme import MetaData
import plotly.graph_objs as go

from base import StreamDeconvolution, StreamInterpolation
from MET4FOFDataReceiver import DataReceiver


class ConnectorAgent(MetrologicalAgent):
    def init_parameters(self, **kwargs):
        super().init_parameters(**kwargs)

        # init connection to smart up unit board
        DR = DataReceiver("127.0.0.1", 7654)
        time.sleep(5)  # wait for sensor Description to be sent

        # get instance of first (and only sensor)
        firstSensorId = list(DR.AllSensors.keys())[0]

        # extract information to set up metadata
        device_id = str(DR.AllSensors[firstSensorId])
        time_name = "time"
        time_unit = "\\second"
        quantity_names = []
        quantity_units = []
        misc = {
            "general": {
                "chid": [],
                "min_scale": [],
                "max_scale": [],
                "resolution": [],
            },
            "calibration": {
                "type": [],
                "parameters": [],
            },
            "location": None,
        }

        channels = DR.AllSensors[firstSensorId].Description.Channels
        for i_channel in channels.keys():

            # read provided description from sensor
            description = channels[i_channel].Description

            quantity_names.append(description["PHYSICAL_QUANTITY"])
            quantity_units.append(description["UNIT"])

            misc["general"]["chid"].append(description["CHID"])
            misc["general"]["min_scale"].append(description["MIN_SCALE"])
            misc["general"]["max_scale"].append(description["MAX_SCALE"])
            misc["general"]["resolution"].append(description["RESOLUTION"])

            # sideload calibration information for X-axis acceleration
            transfer_function_type = None
            transfer_function = None

            if description["PHYSICAL_QUANTITY"] == "X Angular velocity":
                # transfer behavior given by frequency response
                #path = "data/200623_MPU_9250_SN_12_X_Achse_3_COLAREF/200623_MPU_9250_SN_12_X_Achse_3_COLAREFTF.json"
                #transfer_function_type = "frequency_response"

                # transfer behavior given by IIR
                path = "data/200623_MPU_9250_SN_12_X_Achse_3_COLAREF/200623_MPU_9250_SN_12_X_Achse_3_COLAREF_TF_IIR.json"
                transfer_function_type = "continuous_infinite_impulse_response"

                # import
                f = open(path, "r")
                transfer_function = json.load(f)
                f.close()

            misc["calibration"]["type"].append(transfer_function_type)
            misc["calibration"]["parameters"].append(transfer_function)

        meta = MetaData(
            device_id=device_id,
            time_name=time_name,
            time_unit=time_unit,
            quantity_names=quantity_names,
            quantity_units=quantity_units,
            misc=misc,
        )

        # set output description
        self.set_output_data(channel="default", metadata=meta)
        self._output_data["default"]["buffer"].return_type = "list"  # kind of HACK

        # log sensor name
        self.log_info(f"First sensor is {device_id}.")

        # set callback
        DR.AllSensors[firstSensorId].SetCallback(self.sensor_callback)

    def sensor_callback(self, message, description):
        if self.current_state == "Running":
            t = getattr(message, "unix_time") + 1e-9 * getattr(
                message, "unix_time_nsecs"
            )
            ut = 1e-9 * getattr(message, "time_uncertainty")
            v = self.unpack_sensor_data(message)
            uv = np.zeros_like(v)
            data = [[t, ut, v, uv]]

            self.set_output_data(channel="default", data=data)

    def unpack_sensor_data(self, message):
        n_active_channels = len(
            self._output_data["default"]["metadata"].quantities["quantity_names"]
        )

        # collect active indices
        data_np = np.array(
            [
                getattr(message, "Data_{counter:02d}".format(counter=i + 1))
                for i in range(n_active_channels)
            ]
        )
        return data_np


class InputUncertaintyAgent(MetrologicalAgent):

    metadata_is_set = False

    def agent_loop(self):
        if self.current_state == "Running":
            # NOTE: currently supports only interpolation of first _input_data-stream
            if len(self._input_data.keys()) > 0:
                upstream_agent_name = list(self._input_data.keys())[0]

                # generate+set output-metadata on first iteration
                if not self.metadata_is_set:
                    metadata = self._input_data[upstream_agent_name]["metadata"]
                    output_metadata = copy.deepcopy(metadata)
                    misc = copy.deepcopy(metadata.misc)
                    if "general" in misc.keys():
                        misc["general"].pop("min_scale", None)
                        misc["general"].pop("max_scale", None)
                        misc["general"].pop("resolution", None)
                    output_metadata._metadata.update(misc=misc)
                    self.set_output_data(channel="default", metadata=output_metadata)
                    self.metadata_is_set = True

                # pop latest additions to input buffer
                buffer = self._input_data[upstream_agent_name]["buffer"]
                metadata = self._input_data[upstream_agent_name]["metadata"]
                datapoints = buffer.pop(len(buffer))

                if len(datapoints):
                    # apply quantization uncertainty to datapoints
                    datapoints_with_uncertainty = self.add_quantization_unc(
                        datapoints, metadata
                    )

                    # fill output buffer
                    self.set_output_data(
                        channel="default", data=datapoints_with_uncertainty
                    )

        # run the base metrological agent loop
        super().agent_loop()

    def add_quantization_unc(self, datapoints, metadata):
        misc = metadata.misc
        if "general" in misc.keys():

            min_scale = np.array(misc["general"]["min_scale"])
            max_scale = np.array(misc["general"]["max_scale"])
            resolution = np.array(misc["general"]["resolution"])

            # +/- 1/2 LSB
            channel_unc = 0.5 * (max_scale - min_scale) / resolution

            for dp in datapoints:
                dp[3] = dp[3] + channel_unc

        return datapoints


class InterpolationAgent(MetrologicalAgent):

    metadata_is_set = False

    def init_parameters(
        self,
        dt=0.001,
        offset=0.0,
        skipdistance=100,
        kind="linear",
        **kwargs,
    ):
        self.stream_interpolation = StreamInterpolation(
            dt=dt, offset=offset, skipdistance=skipdistance, kind=kind
        )
        super().init_parameters(**kwargs)

    def agent_loop(self):
        if self.current_state == "Running":
            # NOTE: currently supports only interpolation of first _input_data-stream
            if len(self._input_data.keys()) > 0:
                upstream_agent_name = list(self._input_data.keys())[0]

                # generate+set output-metadata on first iteration
                if not self.metadata_is_set:
                    metadata = self._input_data[upstream_agent_name]["metadata"]
                    output_metadata = copy.deepcopy(metadata)
                    misc = copy.deepcopy(metadata.misc)
                    if "modifications" in misc.keys():
                        misc["modifications"].update(
                            **{"interpolated": True, "equidistant": True}
                        )
                    else:
                        misc["modifications"] = {
                            "interpolated": True,
                            "equidistant": True,
                        }
                    output_metadata._metadata.update(misc=misc)
                    self.set_output_data(channel="default", metadata=output_metadata)
                    self.metadata_is_set = True

                # process input stream
                buffer = self._input_data[upstream_agent_name]["buffer"]
                if len(buffer):
                    # pop latest additions to input buffer
                    datapoints = buffer.pop(len(buffer))

                    # apply enhancement to datapoints
                    interpolated_datapoints = (
                        self.stream_interpolation.interpolate_datapoints(datapoints)
                    )

                    # fill output buffer
                    self.set_output_data(
                        channel="default", data=interpolated_datapoints
                    )

        # run the base metrological agent loop
        super().agent_loop()


class DeconvolutionAgent(MetrologicalAgent):
    def init_parameters(self, **kwargs):
        self.metadata_is_set = False
        self.data_is_equidistant = False
        self.stream_deconvolution = None

        super().init_parameters(**kwargs)

    def agent_loop(self):
        if self.current_state == "Running":
            # NOTE: currently supports only process of first _input_data-stream
            if len(self._input_data.keys()) > 0:
                upstream_agent_name = list(self._input_data.keys())[0]

                # generate+set output-metadata on first iteration
                if not self.metadata_is_set:
                    # self.stream_deconvolution = StreamDeconvolution(dt=dt, )

                    metadata = self._input_data[upstream_agent_name]["metadata"]
                    output_metadata = copy.deepcopy(metadata)
                    misc = copy.deepcopy(metadata.misc)
                    if "modifications" in misc.keys():
                        if "equidistant" in misc["modifications"]:
                            self.data_is_equidistant = True
                            misc["modifications"]["deconvolved"] = True

                    if self.data_is_equidistant and "calibration" in misc.keys():
                        transfer_types_per_channel = misc["calibration"]["type"]
                        transfer_parameters_per_channel = misc["calibration"][
                            "parameters"
                        ]

                        self.stream_deconvolution = StreamDeconvolution(
                            transfer_types_per_channel, transfer_parameters_per_channel
                        )

                    output_metadata._metadata.update(misc=misc)
                    self.set_output_data(channel="default", metadata=output_metadata)
                    self.metadata_is_set = True

                # process input stream
                buffer = self._input_data[upstream_agent_name]["buffer"]
                if (
                    isinstance(self.stream_deconvolution, StreamDeconvolution)
                    and self.data_is_equidistant
                    and len(buffer)
                ):
                    # pop latest additions to input buffer
                    datapoints = buffer.pop(len(buffer))

                    # apply enhancement to datapoints
                    corrected_datapoints = (
                        self.stream_deconvolution.deconvolve_datapoints(datapoints)
                    )

                    # fill output buffer
                    self.set_output_data(channel="default", data=corrected_datapoints)

        # run the base metrological agent loop
        super().agent_loop()


class MetrologicalMonitorAgent_multichannel(MetrologicalMonitorAgent):
    def get_data(self, data):
        """Transform list of tuples to four arrays"""

        datapoints = data["data"]
        description = data["metadata"][0]

        t = np.array([dp[0] for dp in datapoints])
        ut = np.array([dp[1] for dp in datapoints])
        v = np.vstack([dp[2] for dp in datapoints])
        uv = np.vstack([dp[3] for dp in datapoints])

        return (t, ut, v, uv), description

    def custom_plot_function(self, data, sender_agent="", **kwargs):
        # TODO: cannot set the label of the xaxis within this method

        traces = []

        # data display
        if "data" in data.keys():
            if len(data["data"]):
                (t, ut, v, uv), desc = self.get_data(data)

                # use description
                t_name, t_unit = desc.time.values()

                for v_channel, uv_channel, v_name_channel, v_unit_channel in zip(
                    v.T,
                    uv.T,
                    desc.quantities["quantity_names"],
                    desc.quantities["quantity_units"],
                ):

                    x_label = f"{t_name} [{t_unit}]"
                    y_label = f"{v_name_channel} [{v_unit_channel}]"

                    trace = go.Scatter(
                        x=t,
                        y=v_channel,
                        error_x=dict(type="data", array=ut, visible=True),
                        error_y=dict(type="data", array=uv_channel, visible=True),
                        mode="lines",
                        name=f"{y_label} ({sender_agent})",
                    )
                    traces.append(trace)
            else:
                traces.append(go.Scatter())
        else:
            traces.append(go.Scatter())

        return traces


class DumpAgent(MetrologicalAgent):

    metadata_is_set = False

    def agent_loop(self):
        if self.current_state == "Running":
            if len(self._input_data.keys()) > 0:
                for upstream_agent_name in self._input_data.keys():

                    # pop latest additions to input buffer
                    buffer = self._input_data[upstream_agent_name]["buffer"]
                    metadata = self._input_data[upstream_agent_name]["metadata"]
                    datapoints = buffer.pop(len(buffer))

                    if len(datapoints):
                        # dump to file
                        self.dump_buffer(upstream_agent_name, datapoints, metadata)

        # run the base metrological agent loop
        super().agent_loop()

    def dump_buffer(self, from_agent, datapoints=None, metadata=None):

        dump_dir = "dump"
        file_dir = f"{dump_dir}/{from_agent}.h5"

        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        # create file if not existing, otherwise load the file
        if not os.path.exists(file_dir):
            f = h5py.File(file_dir, "w")
            group = f.create_group("time_series")
            t_dset = group.create_dataset("t", (0,), dtype=np.float64, maxshape=(None,))
            ut_dset = group.create_dataset(
                "ut", (0,), dtype=np.float64, maxshape=(None,)
            )
            v_dset = group.create_dataset(
                "v",
                (0, 0),
                dtype=np.float64,
                maxshape=(None, None),
            )
            uv_dset = group.create_dataset(
                "uv",
                (0, 0),
                dtype=np.float64,
                maxshape=(None, None),
            )
        else:
            f = h5py.File(file_dir, "r+")
            group = f["time_series"]
            t_dset = group["t"]
            ut_dset = group["ut"]
            v_dset = group["v"]
            uv_dset = group["uv"]

        if metadata is not None:
            for key, val in metadata.metadata.items():
                group.attrs.modify(key, json.dumps(val))

        if datapoints is not None:
            if len(datapoints):
                # read data from datapoints
                t = np.array([dp[0] for dp in datapoints])
                ut = np.array([dp[1] for dp in datapoints])
                v = np.vstack([dp[2] for dp in datapoints])
                uv = np.vstack([dp[3] for dp in datapoints])

                # resize h5-datasets to new length
                t_dset.resize((len(t_dset) + len(t),))
                ut_dset.resize((len(ut_dset) + len(ut),))
                v_dset.resize((len(v_dset) + len(v), v.shape[1]))
                uv_dset.resize((len(uv_dset) + len(uv), uv.shape[1]))

                # write data
                t_dset[-len(t) :] = t
                ut_dset[-len(ut) :] = ut
                v_dset[-len(v) :] = v
                uv_dset[-len(uv) :] = uv

        # end file access
        f.close()

        return None