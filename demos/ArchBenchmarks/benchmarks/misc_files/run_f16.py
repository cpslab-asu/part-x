import logging
import math

import numpy as np
from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from numpy.typing import NDArray
from staliro.models import StaticInput, SignalTimes, SignalValues, ModelData, blackbox
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import staliro
from partx.interfaces.staliro import PartX
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

MAX_BUDGET = 300
NUMBER_OF_MACRO_REPLICATIONS = 50
ALTITUDE = 2338

F16DataT = ModelData[NDArray[np.float_], None]


@blackbox()
def f16_model(static: StaticInput, times: SignalTimes, signals: SignalValues) -> F16DataT:
    power = 9
    alpha = np.deg2rad(2.1215)
    beta = 0
    alt = ALTITUDE
    vel = 540
    phi = static[0]
    theta = static[1]
    psi = static[2]

    initial_state = [vel, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    step = 1 / 30
    autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str = "old")

    result = run_f16_sim(initial_state, max(times), autopilot, step, extended_states=True)
    trajectories: NDArray[np.float_] = result["states"][:, 11:12]
    timestamps: NDArray[np.float_] = result["times"]
    # print(trajectories.T)
    return ModelData(trajectories.T, timestamps)


phi = "always[0, 15] (alt >= 0)"
specification = RTAMTDense(phi, {"alt": 0})

initial_conditions = [
    math.pi / 4 + np.array([-math.pi / 20, math.pi / 30]),  # PHI
    -math.pi / 2 * 0.8 + np.array([0, math.pi / 20]),  # THETA
    -math.pi / 4 + np.array([-math.pi / 8, math.pi / 8]),  # PSI
]



gpr_model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )

optimizer = PartX(
    benchmark_name="F16_budget_{}_altitude{}".format(MAX_BUDGET, ALTITUDE),
    test_function_dimension=3,
    initialization_budget=50,
    continued_sampling_budget=10,
    number_of_BO_samples=[20],
    M=500,
    R=20,
    branching_factor=2,
    alpha=[0.05],
    delta=0.001,
    macro_replication=NUMBER_OF_MACRO_REPLICATIONS,
    seed = 1000,
    fv_quantiles_for_gp=[0.01, 0.05, 0.5],
    results_at_confidence=0.95,
    gpr_params=list(["other", gpr_model]),
    results_folder_name="F16",
    num_cores=10,
)


# Pass to Psy-Taliro and Run
options = Options(runs=1, iterations=MAX_BUDGET, interval=(0, 15), static_parameters=initial_conditions, signals = [])

result = staliro(
            f16_model,
            specification,
            optimizer,
            options
        )