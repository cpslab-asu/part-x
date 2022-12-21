import numpy as np
from numpy.typing import NDArray
from staliro.core.interval import Interval
from staliro.core.model import Model, ModelData, StaticInput, Signals

try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True
SCDataT = NDArray[np.float_]
SCResultT = ModelData[SCDataT, None]


class SCModel(Model[SCResultT, None]):
    MODEL_NAME = "steamcondense_RNN_22"

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

    def simulate(self, static: StaticInput, signals: Signals, intrvl: Interval) -> SCResultT:

        sim_t = matlab.double([0, intrvl.upper])
        n_times = (intrvl.length // self.sampling_step) + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in signals])
        # print(signal_times)
        # print(signal_values)
        # print(signal_times.shape)
        # print(signal_values.shape)
        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())

        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        timestamps_array = np.array(timestamps).flatten()
        data_array = np.array(data)
        # print("*************************")
        # print(timestamps_array)
        # print(timestamps_array.shape)
        # print(diff_array.T.shape)
        return ModelData(data_array.T, timestamps_array)
