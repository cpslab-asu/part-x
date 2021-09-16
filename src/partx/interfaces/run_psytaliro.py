from .run_standalone import run_partx

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
    continued_sampling_budget: int
    number_of_BO_samples: list
    number_of_samples_gen_GP: int
    branching_factor: int
    nugget_mean: float
    nugget_std_dev: float
    alpha: list
    delta: float
    number_of_macro_replications: int
    initial_seed: int

def _optimize(func: OptimizationFn, options: Options, optimizer_options: PartXOptimizerOptions):
    bounds = [bound.astuple() for bound in options.bounds]
    region_support = np.array([list(map(list, bounds))])
    start_time = datetime.now()
    
    results = run_partx(
        benchmark_name=optimizer_options.benchmark_name,
        test_function=func,
        test_function_dimension=optimizer_options.test_function_dimension,
        region_support=region_support,
        initialization_budget=optimizer_options.initialization_budget,
        maximum_budget=options.iterations,
        continued_sampling_budget=optimizer_options.continued_sampling_budget,
        number_of_BO_samples=optimizer_options.number_of_BO_samples,
        number_of_samples_gen_GP=optimizer_options.number_of_samples_gen_GP,
        branching_factor=optimizer_options.branching_factor,
        nugget_mean=optimizer_options.nugget_mean,
        nugget_std_dev=optimizer_options.nugget_std_dev,
        alpha=optimizer_options.alpha,
        delta=optimizer_options.delta,
        number_of_macro_replications=optimizer_options.number_of_macro_replications,
        initial_seed=optimizer_options.initial_seed
    )
    end_time = datetime.now()
    
    return results


def part_x_optimizer(
    func: OptimizationFn,
    options: Options,
    optimizer_options: PartXOptimizerOptions
):
    return _optimize(func, options, optimizer_options)

class PartX(Optimizer[Run]):
    def __init__(self, **kwargs):
        self.optimizer_options = PartXOptimizerOptions(
            benchmark_name=kwargs['benchmark_name'],
            test_function_dimension=kwargs['test_function_dimension'],
            initialization_budget=kwargs['initialization_budget'],
            continued_sampling_budget=kwargs['continued_sampling_budget'],
            number_of_BO_samples=kwargs['number_of_BO_samples'],
            number_of_samples_gen_GP=kwargs['number_of_samples_gen_GP'],
            branching_factor=kwargs['branching_factor'],
            nugget_mean=kwargs['nugget_mean'],
            nugget_std_dev=kwargs['nugget_std_dev'],
            alpha=kwargs['alpha'],
            delta=kwargs['delta'],
            number_of_macro_replications=kwargs['number_of_macro_replications'],
            initial_seed=kwargs['initial_seed']
        )

    def optimize(self, func: OptimizationFn, 
                 options: Options):
        return _optimize(func, options, self.optimizer_options)
