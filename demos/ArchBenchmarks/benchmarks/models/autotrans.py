import numpy as np
from numpy.typing import NDArray
from staliro.core import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace


try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True

AutotransDataT = NDArray[np.float_]
AutotransResultT = ModelResult[AutotransDataT, None]


class AutotransModel(Model[AutotransDataT, None]):
    MODEL_NAME = "Autotrans_shift"

    def __init__(self) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        engine = matlab.engine.start_matlab()
        # engine.addpath("examples")
        model_opts = engine.simget(self.MODEL_NAME)

        self.sampling_step = 0.05
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

    def simulate(self, signals: ModelInputs, intrvl: Interval) -> AutotransResultT:
        sim_t = matlab.double([0, intrvl.upper])
        n_times = (intrvl.length // self.sampling_step) + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in signals.signals])

        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())
        
        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        timestamps_list = np.array(timestamps).flatten().tolist()
        data_list = list(data)
        trace = Trace(timestamps_list, data_list)

        return BasicResult(trace)
