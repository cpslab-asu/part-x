from collections import OrderedDict
from math import pi

from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from numpy import array, deg2rad, float32, float64, ndarray
from staliro.models import blackbox, ModelData


F16_PARAM_MAP = OrderedDict({
    'air_speed': {
        'enabled': False,
        'default': 540
    },
    'angle_of_attack': {
        'enabled': False,
        'default': deg2rad(2.1215)
    },
    'angle_of_sideslip': {
        'enabled': False,
        'default': 0
    },
    'roll': {
        'enabled': True,
        'default': None,
        'range': (pi / 4) + array((-pi / 20, pi / 30)),
    },
    'pitch': {
        'enabled': True,
        'default': None,
        'range': (-pi / 2) * 0.8 + array((0, pi / 20)),
    },
    'yaw': {
        'enabled': True,
        'default': None,
        'range': (-pi / 4) + array((-pi / 8, pi / 8)),
    },
    'roll_rate': {
        'enabled': False,
        'default': 0
    },
    'pitch_rate': {
        'enabled': False,
        'default': 0
    },
    'yaw_rate': {
        'enabled': False,
        'default': 0
    },
    'northward_displacement': {
        'enabled': False,
        'default': 0
    },
    'eastward_displacement': {
        'enabled': False,
        'default': 0
    },
    'altitude': {
        'enabled': False,
        'default': 4040.0
    },
    'engine_power_lag': {
        'enabled': False,
        'default': 9
    }
})


def get_static_params(static_params_map: OrderedDict):
    static_params = []
    for param, config in static_params_map.items():
        if config['enabled']:
            static_params.append(config['range'])
    return static_params


def _compute_initial_conditions(X, param_map):
    conditions = []
    index = 0

    for param, config in param_map.items():
        if config['enabled']:
            conditions.append(X[index])
            index = index + 1
        else:
            conditions.append(config['default'])

    return conditions


@blackbox
def f16_blackbox(X, T, _):
    init_cond = _compute_initial_conditions(X, F16_PARAM_MAP)
    # print(init_cond)
    step = 1 / 30
    autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str="old")

    result = run_f16_sim(init_cond, max(T), autopilot, step, extended_states=True)
    trajectories = result["states"][:, 11].T.astype(float64)
    timestamps = array(result["times"], dtype=(float32))

    return ModelData(trajectories, timestamps)

