from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np
from staliro.core import Interval, Optimizer, ObjectiveFn, Sample

from .run_standalone import run_partx
from .run_standalone_UR import run_partx_UR

Bounds = Sequence[Interval]
PartXResult = List[Any]

@dataclass(frozen=True)
class PartX(Optimizer[PartXResult]):
    """The PartX optimizer provides statistical guarantees about the existence of falsifying behaviour in a system."""

    benchmark_name: str
    test_function_dimension: int
    initialization_budget: int
    continued_sampling_budget: int
    number_of_BO_samples: int
    NGP: Any
    M: Any
    R: Any
    branching_factor: float
    nugget_mean: float
    nugget_std_dev: float
    alpha: float
    delta: float
    macro_replication: int
    fv_quantiles_for_gp: int
    results_at_confidence: float
    results_folder_name: str

    def optimize(self, func: ObjectiveFn, bounds: Bounds, budget:int, seed: int) -> PartXResult:
        region_support = np.array((tuple(bound.astuple() for bound in bounds),))
        print(budget)
        def test_function(sample: np.ndarray) -> float:
            return func.eval_sample(Sample(sample))
        
        return run_partx(
            benchmark_name=self.benchmark_name,
            test_function=test_function,
            test_function_dimension=self.test_function_dimension,
            region_support=region_support,
            initialization_budget=self.initialization_budget,
            maximum_budget=budget,
            continued_sampling_budget=self.continued_sampling_budget,
            number_of_BO_samples=self.number_of_BO_samples,
            NGP=self.NGP,
            M = self.M,
            R = self.R,
            branching_factor=self.branching_factor,
            nugget_mean=self.nugget_mean,
            nugget_std_dev=self.nugget_std_dev,
            alpha=self.alpha,
            delta=self.delta,
            number_of_macro_replications=self.macro_replication,
            initial_seed=seed,
            fv_quantiles_for_gp = self.fv_quantiles_for_gp,
            results_at_confidence = self.results_at_confidence,
            results_folder_name = self.results_folder_name
        )

@dataclass(frozen=True)
class PartXUR(Optimizer[PartXResult]):
    """The PartX Uniform Random optimizer performed PartX optimizations using Uniform Random to select points."""

    benchmark_name: str
    test_function_dimension: int
    macro_replications: int
    results_folder_name: str

    def optimize(self, func: ObjectiveFn, bounds: Bounds, budget: int, seed: int) -> PartXResult:
        region_support = np.array((tuple(bound.astuple() for bound in bounds),))
    
        def test_function(sample: np.ndarray) -> float:
            return func.eval_sample(Sample(sample))

        return run_partx_UR(
            number_of_macro_replications=self.macro_replications,
            benchmark_name=self.benchmark_name,
            initial_seed=seed,
            test_function=test_function,
            test_function_dimension=self.test_function_dimension,
            region_support=region_support,
            results_folder=self.results_folder_name
        )
