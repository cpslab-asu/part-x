
import logging
from .CC_specifications import load_specification_dict
from models import CCModel
from Benchmark import Benchmark

import staliro
from staliro import TestOptions
from partx.gpr import InternalGPR
from partx.bo import InternalBO
from partx.interface import PartX

# logging.basicConfig(level=logging.DEBUG)

# Define Signals and Specification
class Benchmark_CC(Benchmark):
    def __init__(self, benchmark, instance, results_folder) -> None:
        
        self.results_folder = results_folder
        self.specification, self.signals = load_specification_dict(benchmark, instance)
        

        
        self.MAX_BUDGET = 2000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = CCModel()

        
        self.optimizer = PartX(
            BENCHMARK_NAME=f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps",
            oracle_function = None,
            num_macro_reps = self.NUMBER_OF_MACRO_REPLICATIONS,
            init_budget = 20,
            bo_budget = 10,
            cs_budget = 20,
            n_tries_randomsampling = 1,
            n_tries_BO = 1,
            alpha=0.05,
            R = 20,
            M = 500,
            delta=0.001,
            fv_quantiles_for_gp=[0.5,0.05,0.01],
            branching_factor = 2,
            uniform_partitioning = True,
            seed = 12345,
            gpr_model = InternalGPR(),
            bo_model = InternalBO(),
            init_sampling_type = "lhs_sampling",
            cs_sampling_type = "lhs_sampling",
            q_estim_sampling = "lhs_sampling",
            mc_integral_sampling_type = "uniform_sampling",
            results_sampling_type = "uniform_sampling",
            results_at_confidence = 0.95,
            results_folder_name = results_folder,
            num_cores = 1,
        )
        

        self.options = TestOptions(tspan=(0,100), iterations=self.MAX_BUDGET, runs = 1, signals=self.signals)
        
    def run(self):
        result = staliro.test(self.model, self.specification, self.optimizer, self.options)
        # result = staliro(self.model, self.specification, self.optimizer, self.options)