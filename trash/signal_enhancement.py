import time
from typing import Dict
import copy

import numpy as np
from agentMET4FOF.metrological_agents import MetrologicalAgent, MetrologicalMonitorAgent
from time_series_metadata.scheme import MetaData


class SignalEnhancement:

    """Base class to provide utility functions that:
    - transform sensor indications to SI-units
    - enhance measurand estimates with uncertainty (i.e. from datasheets)
    """

    def __init__(self, enhancement_model=None, params=None):
        """Idea:
                            params
                              |
                        ______v______
        ind. val(+unc) | comp. + unc.|
        -------------->|    model    |----> estimated value, uncertainty
                       |             |
                       | (datasheet) |----> output_description
                       |_____________|

        """
        self.params = {}
        self.set_params(params)

        self.output_description = None

        # use if given, otherwise assume no compensation / pass-through
        if enhancement_model is not None:
            self.enhancement_model = enhancement_model
        else:
            self.enhancement_model = (
                lambda indicated_value, indicated_value_unc, params: (
                    indicated_value,
                    indicated_value_unc,
                )
            )

    def enhance(self, indicated_value, indicated_value_unc=0.0):
        """Apply a model/equation to estimate the value of the measurand from an indicated value.
        Also assign an uncertainty based on model or datasheet
        i.e.:
            - translate DAC integer reading to some voltage/acceleration/temperature/...
            - translate percentages to volume according to shape of tank
            - translate unit quantities
        """
        estimated_value, estimated_value_unc = self.enhancement_model(
            indicated_value,
            indicated_value_unc,
            params=self.params,
        )

        return estimated_value, estimated_value_unc

    def set_params(self, new_params, overwrite_all=False):
        """Handle to update/overwrite parameters of the evaluation model.
        By default (replace == False) keys in self.params will be updated from new_params

        :param new_params: dictionary of key-value pairs to add/update in self.params
        :type new_params: dict
        :param overwrite_all: whether to only update keys in params-dict or completely replace old self.params , defaults to False
        :type overwrite_all: bool, optional
        """

        assert isinstance(new_params, dict)

        if overwrite_all:
            self.params = new_params
        else:
            self.params.update(new_params)

    def enhance_timeseries(self, datapoints):
        """Apply specified enhancer-function to
        multiple values comming from time-series-buffer
        at once.
        """

        enhanced_datapoints = []
        for dp in datapoints:
            t = dp[0]
            ut = dp[1]
            x = dp[2]
            ux = dp[3]
            y, uy = self.enhance(x, ux)
            enhanced_datapoints.append((t, ut, y, uy))

        return enhanced_datapoints


class SignalEnhancementAgent(MetrologicalAgent):

    metadata_is_set = False

    def init_parameters(
        self,
        enhancer: SignalEnhancement = None,
        input_data_maxlen=25,
        output_data_maxlen=25,
        **kwargs
    ):

        self._enhancer = enhancer

        super().init_parameters(input_data_maxlen, output_data_maxlen)

    def agent_loop(self):
        if self.current_state == "Running":
            # NOTE: currently supports only enhancement of first _input_data-stream
            if len(self._input_data.keys()) > 0:
                upstream_agent_name = list(self._input_data.keys())[0]

                # generate+set output-metadata on first iteration
                if not self.metadata_is_set:
                    metadata = self._input_data[upstream_agent_name]["metadata"]
                    output_metadata = copy.copy(metadata)
                    output_metadata._metadata.update(self._enhancer.output_description)
                    self.set_output_data(channel="default", metadata=output_metadata)
                    self.metadata_is_set = True

                # pop latest additions to input buffer
                buffer = self._input_data[upstream_agent_name]["buffer"]
                datapoints = buffer.pop(len(buffer))

                # apply enhancement to datapoints
                enhanced_datapoints = self._enhancer.enhance_timeseries(datapoints)

                # fill output buffer
                self.set_output_data(channel="default", data=enhanced_datapoints)

        # run the base metrological agent loop
        super().agent_loop()

    @property
    def metadata(self) -> Dict:
        return self._signal_stream.metadata.metadata