import logging

import numpy as np
from numpy.typing import NDArray
import logging

import numpy as np
from numpy.typing import NDArray
from staliro.core.interval import Interval
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.models import StaticInput, SignalTimes, SignalValues, ModelData, blackbox
from staliro.options import Options, SignalOptions
from staliro.specifications import TLTK, RTAMTDense
from ..models import CCModel
from staliro.staliro import staliro, simulate_model
import scipy.io
# from taliro import tptaliro
###############################################################################
# Define BlackBox Model
# Here, we define the Auto-Transmission Black Box Model from MATLAB into Python

# try:
#     import matlab
#     import matlab.engine
# except ImportError:
#     _has_matlab = False
# else:
#     _has_matlab = True



# CCDataT = NDArray[np.float_]
# CCResultT = ModelData[CCDataT, None]

# class CCModel(Model[CCResultT, None]):
#     MODEL_NAME = "cars"

#     def __init__(self) -> None:
#         if not _has_matlab:
#             raise RuntimeError(
#                 "Simulink support requires the MATLAB Engine for Python to be installed"
#             )

#         engine = matlab.engine.start_matlab()
#         # engine.addpath("examples")
#         model_opts = engine.simget(self.MODEL_NAME)

#         self.sampling_step = 0.05
#         self.engine = engine
#         self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

#     def simulate(self, static: StaticInput, signals: Signals, intrvl: Interval) -> CCResultT:
#         sim_t = matlab.double([0, intrvl.upper])
#         n_times = (intrvl.length // self.sampling_step) + 2
#         signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
#         signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in signals])
        
#         model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())

#         timestamps, _, data = self.engine.sim(
#             self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
#         )

#         print(timestamps)
        
        

#         timestamps_array = np.array(timestamps).flatten()
#         data_array = np.array(data)
#         y54 = (data_array[:,4]-data_array[:,3]).reshape((-1,1))
#         y43 = (data_array[:,3]-data_array[:,2]).reshape((-1,1))
#         y32 = (data_array[:,2]-data_array[:,1]).reshape((-1,1))
#         y21 = (data_array[:,1]-data_array[:,0]).reshape((-1,1))
#         diff_array = np.hstack((y21, y32, y43, y54))
#         # print(diff_array.shape)
#         return ModelData(diff_array.T, timestamps_array)

model = CCModel()


#####################################################################################################################
# Define Signals and Specification

signals = [
    SignalOptions(control_points=[(0, 1)]*10, signal_times=np.linspace(0.,100.,10)),
    SignalOptions(control_points=[(0,1)]*10, signal_times=np.linspace(0.,100.,10)),
]

##################################################
# Need to check with Jacob

# phi = "G[0,65] (F[0,30] (G[0,20] (y54 >= 8)))"
# # specification = MTL.Global(float(0), float(65), MTL.Finally(float(0),float(30), MTL.Global(float(0),float(20), MTL.Predicate("y54", float(-1), float(-8)))))

# spec_tltk = TLTK(phi, {"y54": 3})
# spec_rtamtdense = RTAMTDense(phi, {"y54":3})
# spec_mtl = MTL.Global(float(0), float(65), MTL.Finally(float(0),float(30), MTL.Global(float(0),float(20), MTL.Predicate("y54", float(-1), float(-8)))))
# spec_tptaliro = "@Var_t1"


phi = "G[0,70] (F[0,30] (y54 >= 15))"

specification = RTAMTDense(phi, {"y54":3})
# spec_tltk = TLTK(phi, {"y54":3})
############################################
#######

# options = Options(runs=1, iterations=1, interval=(0, 100), signals=signals)


# def generateRobustnessByMTL(sample, inModel, options: Options, specification):

#     result = simulate_model(inModel, options, sample)
#     # times = states_times
#     # return specification.evaluate(result.states, result.times)
    
#     specification.reset()
#     specification.eval_interval({"y54" : result.states[3,:].astype(np.float64)}, result.times.astype(np.float32))
#     return specification.robustness

# sample1 = [0.23399508, 0.21951849, 0.1530713 , 0.21214886, 0.05941586,
#               0.00148408, 0.05118548, 0.44526711, 0.57185998, 0.70241653,
#               0.61345783, 0.62125773, 0.61758477, 0.52685516, 0.8472919 ,
#               0.73152203, 0.10735846, 0.15916549, 0.79058682, 0.85563365]

# rob1 = generateRobustness(sample1, cc_simulate, options, specification)
# print(rob1)


options = Options(runs=1, iterations=1, interval=(0, 100), signals=signals)


def generateRobustness(sample, inModel, options: Options, specification):

    result = simulate_model(inModel, options, sample)
    # print(result.states.shape)
    # print(result.states)
    # print("***************************************")
    # print(result.times.shape)
    # print(result.times)
    # scipy.io.savemat('states_data.mat', mdict={'arr': result.states})
    # scipy.io.savemat('time_data.mat', mdict={'arr': result.times})
    # times = states_times
    # scipy.io.savemat("CC4_sample2.mat", {"states": result.states, "times": result.times})
    return specification.evaluate(result.states, result.times)

sample1 = [0.90311635, 0.1752506 , 0.93631134, 0.97741106, 0.96674805,
              0.28345545, 0.5548273 , 0.18714629, 0.28300858, 0.48705733,
              0.28222768, 0.99974888, 0.06856977, 0.48588086, 0.14659562,
              0.81915855, 0.64118368, 0.56739029, 0.97795566, 0.19248988 ]

# rob1 = generateRobustness(sample1, model, options, spec_tltk)
rob2 = generateRobustness(sample1, model, options, specification)
# rob3 = generateRobustnessByMTL(sample1, cc_simulate, options, spec_mtl)

# print(f"Rob. by tltk = {rob1}")
print(f"Rob. by rtamtdense = {rob2}")
# print(f"Rob. by MTL = {rob3}")
# print(f"Rob. by MATLAB = 2")