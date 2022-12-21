import logging

import numpy as np
from numpy.typing import NDArray

from staliro.options import Options, SignalOptions
from staliro.specifications import TLTK, RTAMTDense
from ..models import AutotransModel
from staliro.staliro import staliro, simulate_model
import scipy.io
import matplotlib.pyplot as plt
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


# AutotransDataT = NDArray[np.float_]
# AutotransResultT = ModelData[AutotransDataT, None]

# eng = matlab.engine.start_matlab()
# MODEL_NAME = "Autotrans_shift"
# mo = eng.simget(MODEL_NAME)
# model_opts = eng.simset(mo, "SaveFormat", "Array")


# @blackbox
# def at_simulate(
#     static: StaticInput, times: SignalTimes, signals: SignalValues
# ) -> AutotransResultT:


#     sim_t = matlab.double([0, max(times)])
#     model_input = matlab.double(np.row_stack((times, signals)).T.tolist())


#     timestamps, _, data = eng.sim(
#         MODEL_NAME, sim_t, model_opts, model_input, nargout=3
#     )

#     timestamps_array = np.array(timestamps).flatten()
#     data_array = np.array(data)

    
#     return ModelData(data_array.T, timestamps_array)



#####################################################################################################################
# Define Signals and Specification

signals = [
    SignalOptions(control_points = [(0, 100)]*7, signal_times=np.linspace(0.,50.,7)),
    SignalOptions(control_points = [(0, 325)]*3, signal_times=np.linspace(0.,50.,3)),
]

print(signals)
options = Options(runs=1, iterations=1, interval=(0, 50), signals=signals)



phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,8] (speed <= 50)))"
specification_rtamt = RTAMTDense(phi, {"speed":0, "rpm": 1})


def generateRobustness(sample, inModel, options: Options, specification):
    
    result = simulate_model(inModel, options, sample)
    print(result.states)
    print(result.times)
    # plt.plot(result.times, result.states[2,:])
    # plt.show()
    return specification.evaluate(result.states, result.times)

sample1 = [ 53.91035568,  37.62833442,  27.61815572,   8.72383823,
                  43.51217366,  58.6046644 ,  98.69710394,  57.15512412,
                 142.80534066, 284.4460028]

sample2 = [50.1746604 ,  46.86791353,  24.57013138,  29.79803092,
               30.66922006,  13.0928013 ,   0.        ,   0.        ,
               76.59091936, 325.        ]

sample3 = [50.        , 47.94314745, 27.71763691,  0.        ,  0.        ,
              91.2017856 ,  0.        , 11.0411098 , 87.19888107,  0.        ]                

autotrans_blackbox = AutotransModel()
# rob1 = generateRobustness(sample1, autotrans_blackbox, options, specification_rtamt)
# rob2 = generateRobustness(sample2, autotrans_blackbox, options, specification_rtamt)
rob3 = generateRobustness(sample3,  autotrans_blackbox, options, specification_rtamt)


# print(f"Rob. Sample 1 = {rob1}")
# print(f"Rob. Sample 2 = {rob2}")
print(f"Rob. Sample 3 = {rob3}")
# print(str(rob1))

