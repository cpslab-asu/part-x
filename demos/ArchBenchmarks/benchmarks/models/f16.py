import numpy as np
from numpy.typing import NDArray
from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from staliro.core.interval import Interval
from staliro.core.model import Model, ModelData, StaticInput, Signals

F16DataT = NDArray[np.float_]
F16ResultT = ModelData[F16DataT, None]


class F16Model(Model[F16ResultT, None]):
    def __init__(self, static_params_map) -> None:
        self.F16_PARAM_MAP = static_params_map


    def get_static_params(self):
        static_params = []
        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                static_params.append(config['range'])
        return static_params


    def _compute_initial_conditions(self, X):
        conditions = []
        index = 0

        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                conditions.append(X[index])
                index = index + 1
            else:
                conditions.append(config['default'])

        return conditions

    def simulate(
        self, static: StaticInput, signals: Signals, intrvl: Interval
    ) -> F16ResultT:
        
        init_cond = self._compute_initial_conditions(static)
        
        step = 1 / 30
        autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str="old")

        result = run_f16_sim(init_cond, intrvl.upper, autopilot, step, extended_states=True)
        trajectories = result["states"][:, 11:12].T.astype(np.float64)

        timestamps = np.array(result["times"], dtype=(np.float32))

        return ModelData(trajectories, timestamps)