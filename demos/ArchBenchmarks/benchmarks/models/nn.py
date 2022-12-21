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
NNDataT = NDArray[np.float_]
NNResultT = ModelData[NNDataT, None]


class NNModel(Model[NNResultT, None]):
    MODEL_NAME = "narmamaglev_v1"

    def __init__(self, alpha_value, beta_value) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        engine = matlab.engine.start_matlab()
        # engine.addpath("examples")
        model_opts = engine.simget(self.MODEL_NAME)
        self.alpha_value = alpha_value
        self.beta_value = beta_value
        self.sampling_step = 1
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

    def simulate(
        self, static: StaticInput, signals: Signals, intrvl: Interval
    ) -> NNResultT:
        
        sim_t = matlab.double([0, intrvl.upper])
        n_times = intrvl.length // self.sampling_step + 1
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[static[0], static[1], static[2], static[2]]])
        
        self.engine.workspace["u_ts"] = 0.001
        model_input = matlab.double(
            np.row_stack((signal_times, signal_values)).T.tolist()
        )

        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        timestamps_array = np.array(timestamps).flatten()
        data_array = np.array(data)
        
        pos = data_array[:,0]

        pos_ref_mod = np.absolute(data_array[:,0] - data_array[:,2])
        mod_ref = np.absolute(data_array[:,2])

        p1 = pos_ref_mod - self.alpha_value - (self.beta_value * mod_ref)
        p2 = self.alpha_value + (self.beta_value * mod_ref) - pos_ref_mod

        sim_data = np.vstack([pos, p1, p2])
        
        
        return ModelData(sim_data, timestamps_array)