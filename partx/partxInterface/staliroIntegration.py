from dataclasses import dataclass
from typing import Any, List, Sequence, Callable

import numpy as np
from staliro.core import Interval, Optimizer, ObjectiveFn, Sample

from .runMultipleReps import run_partx

Bounds = Sequence[Interval]
PartXResult = List[Any]

@dataclass(frozen=True)
class PartX(Optimizer[PartXResult]):
    """The PartX optimizer provides statistical guarantees about the existence of falsifying behaviour in a system."""

    BENCHMARK_NAME: str
    num_macro_reps: int
    init_budget: int
    bo_budget: int
    cs_budget: int
    alpha: float
    R: int
    M: int
    delta: float
    fv_quantiles_for_gp: float
    branching_factor: int
    uniform_partitioning: bool
    seed: int
    gpr_model: Callable
    bo_model: Callable
    init_sampling_type: str
    cs_sampling_type: str
    q_estim_sampling: str
    mc_integral_sampling_type: str
    results_sampling_type: str
    results_at_confidence: list
    results_folder_name: str
    num_cores: int

    def optimize(self, func: ObjectiveFn, bounds: Bounds, budget:int, seed: int) -> PartXResult:
        region_support = np.array((tuple(bound.astuple() for bound in bounds),))[0]

        print("************************************************************************")
        print("************************************************************************")
        print("************************************************************************")
        print(f"Test Function:\n Testing function is a {region_support.shape[0]}d problem with initial region support of {region_support}.")
        print(f"Starting {self.num_macro_reps} macro replications with maximum budget of {budget}, where")
        print(f"initilization budget = {self.init_budget},\nbo budget = {self.bo_budget},\ncontinued sampling budget = {self.cs_budget}")
        print(f"Sampling Types\n-----------")
        print(f"init_sampling_type = {self.init_sampling_type}")
        print(f"cs_sampling_type = {self.cs_sampling_type}")
        print(f"q_estim_sampling = {self.q_estim_sampling}")
        print(f"mc_integral_sampling_type = {self.mc_integral_sampling_type}")
        print(f"results_sampling_type = {self.results_sampling_type}")
        print("************************************************************************")
        print("************************************************************************")
        print("************************************************************************")
        
        
        def test_function(sample: np.ndarray) -> float:
            return func.eval_sample(Sample(sample))
        
        return run_partx(
            BENCHMARK_NAME = self.BENCHMARK_NAME,
            test_function = test_function,
            num_macro_reps = self.num_macro_reps,
            init_reg_sup = region_support,
            tf_dim = region_support.shape[0],
            max_budget = budget,
            init_budget = self.init_budget,
            bo_budget = self.bo_budget,
            cs_budget = self.cs_budget,
            alpha = self.alpha,
            R = self.R,
            M = self.M,
            delta = self.delta,
            fv_quantiles_for_gp = self.fv_quantiles_for_gp,
            branching_factor = self.branching_factor,
            uniform_partitioning = self.uniform_partitioning,
            start_seed = self.seed,
            gpr_model = self.gpr_model,
            bo_model = self.bo_model,
            init_sampling_type = self.init_sampling_type,
            cs_sampling_type = self.cs_sampling_type, 
            q_estim_sampling = self.q_estim_sampling,
            mc_integral_sampling_type = self.mc_integral_sampling_type,
            results_sampling_type = self.results_sampling_type,
            results_at_confidence = self.results_at_confidence,
            results_folder_name = self.results_folder_name,
            num_cores = self.num_cores
        )