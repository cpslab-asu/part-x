import logging
from unittest import result

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from numpy.typing import NDArray
from staliro.core.interval import Interval
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.models import StaticInput, SignalTimes, SignalValues, ModelData, blackbox
from partx.interfaces.staliro import PartX
from staliro.options import Options, SignalOptions
from staliro.specifications import RTAMTDense
from staliro.staliro import staliro, simulate_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from staliro.signals import piecewise_constant
import scipy.io
###############################################################################
# Define BlackBox Model
# Here, we define the Auto-Transmission Black Box Model from MATLAB into Python


try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True


WTDataT = NDArray[np.float_]
WTResultT = ModelData[WTDataT, None]

class AFCModel(Model[WTResultT, None]):
    MODEL_NAME = "WindTurbine/SimplifiedWTModelFALS.slx"

    def __init__(self) -> None:
        

        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        engine = matlab.engine.start_matlab()
        engine.init_SimpleWindTurbine()

        self.sampling_step = 0.01 
                
        self.engine = engine
        

    def simulate(
        self, static: StaticInput, signals: Signals, intrvl: Interval
    ) -> WTResultT:
        
        
        
        sim_t = matlab.double([0, intrvl.upper])
        n_times = intrvl.length // self.sampling_step + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        in_1 = (signals[0].at_times(signal_times))
        in_2 = (signals[1].at_times(signal_times))
        in_3 = (signals[2].at_times(signal_times))
        in_4 = (signals[3].at_times(signal_times))
        in_5 = (signals[4].at_times(signal_times))
        in_6 = (signals[5].at_times(signal_times))
        
        model_input = matlab.double(np.row_stack((signal_times, in_1, in_2, in_3, in_4, in_5, in_6)).T.tolist())
        
        timestamps, _, data = self.engine.sim(
        self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )
        timestamps_array = np.array(timestamps).flatten()
        data_array = np.array(data)

        return ModelData(data_array.T, timestamps_array)
        
        
        


model = AFCModel()
#####################################################################################################################
# Define Signals and Specification



signals = [
    SignalOptions(control_points=[(900, 1100)], factory=piecewise_constant),
]


# mod_u = "(ut <= 0.007) and (ut >= -0.007)"
# phi = f"G[11,50] ({mod_u})"
# specification = RTAMTDense(phi, {"ut": 0})

# #####################################################################################################################
# # Define Optimizer Options

# MAX_BUDGET = 300
# NUMBER_OF_MACRO_REPLICATIONS = 50


# gpr_model = GaussianProcessRegressor(
#             kernel=Matern(nu=2.5),
#             alpha=1e-6,
#             normalize_y=True,
#             n_restarts_optimizer=5
#         )

# optimizer = PartX(
#     benchmark_name="AFC29_budget_{}".format(MAX_BUDGET),
#     test_function_dimension=11,
#     initialization_budget=50,
#     continued_sampling_budget=10,
#     number_of_BO_samples=[20],
#     M=500,
#     R=20,
#     branching_factor=2,
#     alpha=[0.05],
#     delta=0.001,
#     macro_replication=NUMBER_OF_MACRO_REPLICATIONS,
#     seed = 1000,
#     fv_quantiles_for_gp=[0.01, 0.05, 0.5],
#     results_at_confidence=0.95,
#     gpr_params=list(["other", gpr_model]),
#     results_folder_name="AFCModel",
#     num_cores=1,
# )

# #####################################################################################################################

# # Pass to Psy-Taliro and Run

# options = Options(runs=1, iterations=MAX_BUDGET, interval=(0, 50),  signals=signals)
# result = staliro(model, specification, optimizer, options)


def generateRobustness(sample, inModel, options: Options, specification):

    return simulate_model(inModel, options, sample)
    # return specification.evaluate(result.states, result.times)

options = Options(runs=1, iterations=1, interval=(0, 50), signals=signals)
    
sample = [920.6589,
   16.4824,
   13.2419,
   15.7851,
   32.8663,
   21.8393,
   47.3608,
   17.8078,
   15.4506,
   22.9059,
   25.6647]


rob = generateRobustness(sample, model, options, specification)
print(rob)

# # trace_matlab = scipy.io.loadmat("afc_trace.mat")['ans']


