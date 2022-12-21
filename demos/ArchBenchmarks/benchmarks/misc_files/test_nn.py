import logging

import numpy as np
from numpy.typing import NDArray
import logging

from ..models import NNModel

from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model

###############################################################################
# Define BlackBox Model
# Here, we define the Auto-Transmission Black Box Model from MATLAB into Python

model = NNModel(0.005, 0.03)
#####################################################################################################################
# Define Signals and Specification

initial_conditions = [
    np.array([1.95, 2.05]),
    np.array([1.95, 2.05]),
    np.array([1.95, 2.05]),
]


options = Options(
    runs=1,
    iterations=1,
    interval=(0, 3),
    static_parameters=initial_conditions,
    signals=[],
)


##################################################
# Define Specification
phi_1 = "F[0,1] (pos >= 3.2)"
phi_2 = "F[1,1.5] (G[0,0.5]((pos >= 1.75) and (pos <= 2.25)))"
phi_3 = "G[2,3] ((pos >= 1.825) and (pos <= 2.175))"
phi = f"(({phi_1}) and ({phi_2}) and ({phi_3}))"
specification = RTAMTDense(phi, {"pos": 0})
##################################################


def generateRobustness(sample, inModel, options: Options, specification):

    result = simulate_model(inModel, options, sample)
    # times = states_times
    return specification.evaluate(result.states, result.times)


sample = [1.96254851, 2.032458885, 2.012548491]

rob = generateRobustness(sample, model, options, specification)
print(rob)
