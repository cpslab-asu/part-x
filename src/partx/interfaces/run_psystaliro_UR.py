from .run_standalone_UR import run_partx_UR
from ..executables.exp_statistics import get_true_fv
from ..executables.generate_statistics import generate_statistics
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from staliro.options import Options
from staliro.optimizers import OptimizationFn, Optimizer
from staliro.results import Result, Run


@dataclass
class PartXOptimizerOptions:
    benchmark_name: str
    test_function_dimension: str
    initialization_budget: int
    number_of_samples: int
    number_of_macro_replications: int
    initial_seed: int
    results_folder: str


benchmark_name, start_seed, test_function, 
                test_function_dimension, region_support,
                number_of_samples, results_folder
def _optimize(func: OptimizationFn, options: Options, optimizer_options: PartXOptimizerOptions):
    bounds = [bound.astuple() for bound in options.bounds]
    region_support = np.array([list(map(list, bounds))])
    start_time = datetime.now()
    
    results = run_partx_UR(
        number_of_macro_replications=optimizer_options.number_of_macro_replications,
        initial_seed=optimizer_options.initial_seed,
        test_function=func,
        test_function_dimension=optimizer_options.test_function_dimension,
        region_support=region_support,
        number_of_samples = optimizer_options.number_of_samples,
        results_folder = optimizer_options.results_folder,
        benchmark_name=optimizer_options.benchmark_name,
    )
    end_time = datetime.now()
    return results


def part_x_optimizer(
    func: OptimizationFn,
    options: Options,
    optimizer_options: PartXOptimizerOptions
):
    return _optimize(func, options, optimizer_options)

class PartX_UR(Optimizer[Run]):
    def __init__(self, **kwargs):
        self.optimizer_options = PartXOptimizerOptions(
            number_of_macro_replications=kwargs['number_of_macro_replications'],
            initial_seed=kwargs['initial_seed'],
            test_function=kwargs['test_function'],
            test_function_dimension=kwargs['test_function_dimension'],
            region_support=kwargs['region_support'],
            number_of_samples = kwargs['number_of_samples'],
            results_folder = kwargs['results_folder'],
            benchmark_name=kwargs['benchmark_name'],
        )

    def optimize(self, func: OptimizationFn, 
                 options: Options):
        return _optimize(func, options, self.optimizer_options)
