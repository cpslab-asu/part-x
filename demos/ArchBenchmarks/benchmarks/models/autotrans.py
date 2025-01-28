

from staliro import Sample, SignalInput, TestOptions, staliro
from staliro.models import Model, Result
from staliro.specifications import rtamt

import numpy as np
from numpy.typing import NDArray
try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True
 
# AutotransDataT = NDArray[np.float_]
# AutotransResultT = Result[AutotransDataT, AutotransDataT]
 
 
class AutotransModel(Model[list[float], None]):
    MODEL_NAME = "Autotrans_shift"
 
    def __init__(self) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )
 
        engine = matlab.engine.start_matlab()
        model_opts = engine.simget(self.MODEL_NAME)
        self.sampling_step = 0.05
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")
        print("Model Initialized")

    def simulate(self, sample: Sample) -> Result[list[float], tuple]:

        tstart, tend = sample.signals.tspan
        duration = tend - tstart
        sim_t = matlab.double([0, tend])
        n_times = duration // self.sampling_step
        signal_times = np.linspace(tstart, tend, num=int(n_times))
        signal_values = np.array(
            [[signal.at_time(t) for t in signal_times] for signal in sample.signals]
        )

        # tstart, tend = sample.signals.tspan
        # duration = tend - tstart
        # sim_t = matlab.double([0, tend])
        # n_times = (intrvl.length // self.sampling_step) + 2
        # signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        # signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in signals.signals])
 
        # model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())
        # timestamps, _, data = self.engine.sim(
        #     self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        # )

        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())
        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        times: list[float] = np.array(timestamps).flatten().tolist()
        states: list[list[float]] = list(data)
        extra_inp_times = signal_times
        extra_inp_vals = signal_values
        
        return Result(times=times, states=states, extra=(extra_inp_times, extra_inp_vals))
