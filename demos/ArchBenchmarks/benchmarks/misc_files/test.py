import logging

import numpy as np
from numpy.typing import NDArray
import logging

import numpy as np
from numpy.typing import NDArray
from staliro.core.interval import Interval
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.options import Options, SignalOptions
from staliro.specifications import TLTK
from staliro.staliro import staliro, simulate_model
from staliro.models import StaticInput, SignalTimes, SignalValues, ModelData, blackbox

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


CCDataT = NDArray[np.float_]
CCResultT = ModelData[CCDataT, None]
eng = matlab.engine.start_matlab()
MODEL_NAME = "cars"
mo = eng.simget(MODEL_NAME)
model_opts = eng.simset(mo, "SaveFormat", "Array")


@blackbox
def cc_simulate(
    static: StaticInput, times: SignalTimes, signals: SignalValues
) -> CCResultT:

    # n_times = (max(times) // self.sampling_step) + 2
    # signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
    # signal_values = np.array(
    #     [[signal.at_time(t) for t in signal_times] for signal in signals]
    # )
    # print(signal_times.shape)
    # print(signal_values.shape)
    # with matlab.engine.start_matlab() as eng:
    # print(times)
    # print("*****************")
    # print(signals)
    # print("*****************")
    # print(times.shape)
    # print(signals.shape)
    # print("************************")
    # print(efeof)

    sim_t = matlab.double([0, max(times)])
    model_input = matlab.double(np.row_stack((times, signals)).T.tolist())

    timestamps, _, data = eng.sim(MODEL_NAME, sim_t, model_opts, model_input, nargout=3)

    timestamps_array = np.array(timestamps).flatten()
    data_array = np.array(data)

    y54 = (data_array[:, 4] - data_array[:, 3]).reshape((-1, 1))
    y43 = (data_array[:, 3] - data_array[:, 2]).reshape((-1, 1))
    y32 = (data_array[:, 2] - data_array[:, 1]).reshape((-1, 1))
    y21 = (data_array[:, 1] - data_array[:, 0]).reshape((-1, 1))
    diff_array = np.hstack((y21, y32, y43, y54))
    # print(diff_array.shape)
    return ModelData(diff_array.T, timestamps_array)


#####################################################################################################################
# Define Signals and Specification

##################################################
# Need to check with Jacob

signals = [
    SignalOptions((0, 1), control_points=10, signal_times=np.linspace(0.0, 100.0, 10)),
    SignalOptions((0, 1), control_points=10, signal_times=np.linspace(0.0, 100.0, 10)),
]

phi = "always[0, 50] (y54 <= 40)"
specification = TLTK(phi, {"y54": 3})
##################################################

options = Options(runs=1, iterations=1, interval=(0, 100), signals=signals)


def generateRobustness(sample, inModel, options: Options, specification):

    result = simulate_model(inModel, options, sample)
    print("Res ready")
    # times = states_times
    return specification.evaluate(result.states, result.times)


print("hehe")
sample = [
    0.82303398,
    0.92704969,
    0.5412947,
    0.66717901,
    0.62367045,
    0.72994748,
    0.88056504,
    0.55457143,
    0.20092732,
    0.98563532,
    0.66147961,
    0.41065468,
    0.02204427,
    0.40719673,
    0.69690379,
    0.65193042,
    0.78798888,
    0.55421307,
    0.34035913,
    0.69069475,
]

rob = generateRobustness(sample, cc_simulate, options, specification)
print(rob)
print(10.096424102783203)
print("done")
