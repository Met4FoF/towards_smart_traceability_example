import numpy as np
from PyDynamic.model_estimation import invLSIIR_unc
from PyDynamic.uncertainty.interpolate import interp1d_unc
from PyDynamic.uncertainty.propagate_DFT import AmpPhase2DFT
from PyDynamic.uncertainty.propagate_filter import IIRuncFilter
from scipy import signal
from scipy.signal import sawtooth
from time_series_buffer import TimeSeriesBuffer


class StreamOperation:
    def datapoints_to_arrays(self, datapoints):
        """Transform list of tuples to four arrays"""
        t_in = np.array([dp[0] for dp in datapoints])
        ut_in = np.array([dp[1] for dp in datapoints])
        v_in = np.array([dp[2] for dp in datapoints])
        uv_in = np.array([dp[3] for dp in datapoints])

        return t_in, ut_in, v_in, uv_in

    def arrays_to_datapoints(self, t_out, ut_out, v_out, uv_out):
        """Transform four arrays to list of tuples"""
        datapoints = [
            [t, ut, v, uv] for t, ut, v, uv in zip(t_out, ut_out, v_out, uv_out)
        ]

        return datapoints


class StreamInterpolation(StreamOperation):
    def __init__(self, dt, offset=0.0, skipdistance=None, kind="linear"):
        """
        Initialize a StreamInterpolation class, by defining:

        Parameters
        ----------
        dt : float
            wanted time-spacing at the output
        offset : float, (defaults to 0.0)
            offset to time in unix-second
            e.g. dt=0.2 and offset=0.1 returns output timestamps ending with .1 .3 .5 .7 .9
        skipdistance : float, (defaults to None)
            if two consecutive datapoints are more than skipdistance away from eachother,
            then do not interpolate between those values
        kind : string, (defaults to 'linear')
            referes to interpolation method used by underlying PyDynamic interp1d_unc method

        """

        self.dt = dt
        self.skipdistance = skipdistance
        self.offset = offset
        self.kind = kind

        self.state = {
            "previous_last_input_datapoint": None,
            "previous_last_output_datapoint": None,
        }

    def interpolate_datapoints(self, datapoints):
        """ Interpolate datapoints of time-series """

        # append previous last value
        last_in = self.state["previous_last_input_datapoint"]
        if last_in is not None:
            datapoints = np.insert(datapoints, 0, last_in, axis=0)

        # reshape datapoints to fit input-requirements of interpolation
        t_in, ut_in, v_in, uv_in = self.datapoints_to_arrays(datapoints)

        if len(t_in):
            # calculate t_out
            t_out = self.calculate_output_timestamps(t_in)

            # interpolate (every channel on its own)
            t_out, ut_out, v_out, uv_out = self.interpolate(t_out, t_in, v_in, uv_in)

            # reshape arrays to datapoints
            interpolated_datapoints = self.arrays_to_datapoints(
                t_out, ut_out, v_out, uv_out
            )

            # update internal state
            self.update_internal_state(datapoints, interpolated_datapoints)

        else:
            interpolated_datapoints = [[]]

        return interpolated_datapoints

    def interpolate(self, t_out, t_in, v_in, uv_in):

        # init
        ut_out = np.zeros_like(t_out)  # assume zero time uncertainty of time stamps
        v_out = np.array([])
        uv_out = np.array([])

        if len(t_in) and len(t_out):
            if len(v_in.shape) == 1:
                _, v_out, uv_out = interp1d_unc(
                    t_out, t_in, v_in, uv_in, kind=self.kind, bounds_error=False
                )

            elif len(v_in.shape) == 2:

                v_out_list = []
                uv_out_list = []

                for v_in_channel, uv_in_channel in zip(v_in.T, uv_in.T):
                    _, v_out, uv_out = interp1d_unc(
                        t_out,
                        t_in,
                        v_in_channel,
                        uv_in_channel,
                        kind=self.kind,
                        bounds_error=False,
                    )

                    v_out_list.append(v_out)
                    uv_out_list.append(uv_out)

                # reshape
                v_out = np.vstack(v_out_list).T
                uv_out = np.vstack(uv_out_list).T

            else:
                raise ValueError(
                    "Input shape not supported. Values at a single time-step can be 1D or 2D."
                )

        return t_out, ut_out, v_out, uv_out

    def calculate_output_timestamps(self, t_in):

        last_out = self.state["previous_last_output_datapoint"]
        last_in = self.state["previous_last_input_datapoint"]

        if last_in is not None:
            start_at = last_in[0]
        elif len(t_in):
            start_at = t_in[0]
        else:
            return np.array([])

        t_out_start = self.next_valid_output_timestamp(start_at)

        # assert monotonic time
        if last_out is not None:
            if t_out_start <= last_out[0]:
                raise UserWarning(
                    "Provided time-series does not seem to provide monotonic time. Handle results with caution."
                )

        # TODO: filter t_out if skipdistance is violated
        if np.any(np.diff(t_in) > self.skipdistance):
            raise NotImplementedError("skipdistance is not implemented yet.")

        # define timestamps, at which to interpolate
        t_out = np.arange(t_out_start, t_in[-1], step=self.dt)

        return t_out

    def update_internal_state(self, input_datapoints, output_datapoints):

        # update internal state
        if len(input_datapoints):
            self.state["previous_last_input_datapoint"] = input_datapoints[-1]

        if len(output_datapoints):
            self.state["previous_last_output_datapoint"] = output_datapoints[-1]

    def next_valid_output_timestamp(self, t):
        """Return the next valid timestamp to interpolate to, given dt and offset. """
        t_valid = self.dt * (t // self.dt + 1.0) + self.offset

        return t_valid


class StreamDeconvolution(StreamOperation):
    """Correct a time series by deconvolving it with a suitable
    IIR-filter, which inverts the calibrated transfer behavior.

    The time series might have vector-values entries/channels. It is then assumed
    that every channel has its own transfer behavior.
    """

    def __init__(self, transfer_types, transfer_parameters, dt=0.001, Nb=3, Na=2):
        """
        Initiate a deconvolution of a time series stream. The filter accomplishing
        the deconvolution is calculated from calibration data.

        """
        self.dt = dt
        self.Nb = Nb
        self.Na = Na
        self.parameters = []
        self.states = []

        for tt, tp in zip(transfer_types, transfer_parameters):

            if tt == "frequency_response":
                a, b, Uab = self.frequency_response_correction_filter(
                    tp, self.dt, self.Nb, self.Na
                )
                params = {"a": a, "b": b, "Uab": Uab}

            elif tt == "continuous_infinite_impulse_response":
                a, b, Uab = self.cont_iir_response_correction_filter(
                    tp, self.dt, self.Nb, self.Na
                )
                params = {"a": a, "b": b, "Uab": Uab}

            elif tt == None:
                params = None

            else:
                raise NotImplementedError(
                    f"The given transfer behavior type '{tt}' is not supported (yet)."
                    "Only None and 'frequency_response' or supported."
                )

            self.parameters.append(params)
            self.states.append(None)

    def frequency_response_correction_filter(self, transfer_parameter, dt, Nb, Na):

        F = np.array(transfer_parameter["Frequencys"])
        A = np.array(transfer_parameter["AmplitudeCoefficent"])
        UA = np.array(transfer_parameter["AmplitudeCoefficentUncer"])
        P = np.unwrap(np.array(transfer_parameter["Phase"]))
        UP = np.array(transfer_parameter["PhaseUncer"])
        TAU = np.array(transfer_parameter["Tau"])

        UAP = np.diag(np.r_[UA, UP])

        H, UH = AmpPhase2DFT(A, P, UAP)
        H = H[: len(F)] + 1j * H[len(F) :]

        # estimate IIR
        fs = 1.0 / dt

        b, a, tau, Uab = invLSIIR_unc(H, UH, Nb, Na, F, Fs=fs, tau=np.median(TAU))

        return a, b, Uab

    def cont_iir_response_correction_filter(self, transfer_parameter, dt, Nb, Na, tau=2.0):

        b_given = transfer_parameter["NumeratorCoefficient"]
        ub_given = transfer_parameter["NumeratorCoefficientUncer"]
        a_given = transfer_parameter["DenominatorCoefficient"]
        ua_given = transfer_parameter["DenominatorCoefficientUncer"]
        cr = transfer_parameter["CalibratedRange"]
        cru = transfer_parameter["CalibratedRangeUnit"]

        # generate frequency response with uncertainty via Monte Carlo
        mc_runs = 200
        F = np.linspace(cr[0], cr[1], num=20)

        AA = np.random.multivariate_normal(
            a_given, np.diag(np.square(ua_given)), size=mc_runs
        )
        BB = np.random.multivariate_normal(
            b_given, np.diag(np.square(ub_given)), size=mc_runs
        )
        HH = np.empty((mc_runs, len(F)), dtype=np.complex)

        for aa, bb, i in zip(AA, BB, range(mc_runs)):
            _, HH[i, :] = signal.freqs(*signal.normalize(bb, aa), worN=F)

        # 
        H = np.mean(HH, axis=0)
        UH = np.cov(np.hstack((np.real(HH), np.imag(HH))).T)

        b, a, tau, Uab = invLSIIR_unc(H, UH, Nb, Na, F, Fs=1.0/dt, tau=tau)

        return a, b, Uab

    def deconvolve_datapoints(self, datapoints):
        """ Deconvolve datapoints of time-series """

        # reshape datapoints to fit input-requirements of filter
        t_in, ut_in, v_in, uv_in = self.datapoints_to_arrays(datapoints)

        if len(t_in):
            # deconvolve (every channel on its own)
            v_out, uv_out = self.deconvolve(v_in, uv_in)

            # reshape arrays to datapoints
            deconvolved_datapoints = self.arrays_to_datapoints(
                t_in, ut_in, v_out, uv_out
            )

        else:
            deconvolved_datapoints = [[]]

        return deconvolved_datapoints

    def deconvolve(self, v_in, uv_in):

        # TODO: also shift time

        # init
        v_out = np.array([])
        uv_out = np.array([])

        if len(v_in):
            if len(v_in.shape) == 1:
                if not self.parameters[0] is None:
                    v_out, uv_out, self.states[0] = IIRuncFilter(
                        x=v_in,
                        Ux=uv_in,
                        **self.parameters[0],
                        state=self.states[0],
                        kind="diag",
                    )
                else:
                    v_out = np.full_like(v_in, np.nan)
                    uv_out = np.full_like(v_in, np.nan)

            elif len(v_in.shape) == 2:

                v_out_list = []
                uv_out_list = []

                for i, (v_in_channel, uv_in_channel) in enumerate(zip(v_in.T, uv_in.T)):
                    if not self.parameters[i] is None:
                        v_out, uv_out, self.states[i] = IIRuncFilter(
                            x=v_in_channel,
                            Ux=uv_in_channel,
                            **self.parameters[i],
                            state=self.states[i],
                            kind="diag",
                        )
                    else:
                        v_out = np.full_like(v_in_channel, np.nan)
                        uv_out = np.full_like(v_in_channel, np.nan)

                    v_out_list.append(v_out)
                    uv_out_list.append(uv_out)

                # reshape
                v_out = np.vstack(v_out_list).T
                uv_out = np.vstack(uv_out_list).T

            else:
                raise ValueError(
                    "Input shape not supported. Values at a single time-step can be 1D or 2D."
                )

        return v_out, uv_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dt = 0.001
    t_last = 0.032342

    buf = TimeSeriesBuffer(maxlen=1000, return_type="list")
    buf_interp = TimeSeriesBuffer(maxlen=50, return_type="list")
    si = StreamInterpolation(dt=0.1, offset=0.0, skipdistance=0.5, kind="linear")

    for i in range(250):

        # length of current stream
        if i > 0:
            n = np.random.randint(0, 10)
        else:
            n = 0

        # generate timestamps with slight variation
        if n:
            r = np.random.uniform(-1, 1, n)
            t = t_last + np.cumsum(dt * (1 + 0.1 * r))
            t_last = t[-1]
        else:
            t = np.array([])

        # generate values with uncertainty for each timestamp
        s = 0.02
        v = sawtooth(np.pi * t, 0.7) + s * np.random.randn(n)
        uv = s * (1 + np.abs(v))
        v = np.add.outer(v, np.arange(5))
        uv = np.multiply.outer(uv, np.arange(5))

        if len(t):
            buf.add(time=t, val=v, val_unc=uv)

        # run the interpolation
        datapoints = buf.show(-1)
        interpolated_datapoints = si.interpolate_datapoints(datapoints)
        if len(datapoints):
            buf_interp.add(data=interpolated_datapoints)

    # visualize output
    dps = buf.show(-1)
    idps = buf_interp.show(-1)

    t = np.array([dp[0] for dp in dps])
    v = np.array([dp[2] for dp in dps])
    uv = np.array([dp[3] for dp in dps])

    ti = np.array([dp[0] for dp in idps])
    vi = np.array([dp[2] for dp in idps])
    uvi = np.array([dp[3] for dp in idps])

    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].plot(t, v, "k")
    ax[1].plot(t, uv, "k")

    ax[0].plot(ti, vi, "-or")
    ax[1].plot(ti, uvi, "-or")
    plt.show()
