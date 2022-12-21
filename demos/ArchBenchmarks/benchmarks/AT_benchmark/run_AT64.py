
from AT_benchmark.AT_specifications import load_specification_dict
from models import AutotransModel
from Benchmark import Benchmark
from partx.partxInterface.staliroIntegration import PartX
from partx.bayesianOptimization.internalBO import InternalBO
from partx.gprInterface.internalGPR import InternalGPR

from staliro.staliro import staliro
from staliro.options import Options

# Define Signals and Specification
class Benchmark_AT64(Benchmark):
    def __init__(self, benchmark, results_folder) -> None:
        if benchmark != "AT64":
            raise ValueError("Inappropriate Benchmark name")

        self.results_folder = results_folder
        self.specification, self.signals = load_specification_dict(benchmark)
        print(self.specification)
        print(self.signals)
        self.MAX_BUDGET = 2000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = AutotransModel()
        self.optimizer = PartX(
            BENCHMARK_NAME=f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps",
            num_macro_reps = self.NUMBER_OF_MACRO_REPLICATIONS,
            init_budget = 20,
            bo_budget = 10,
            cs_budget = 20,
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

        self.options = Options(runs=1, iterations=self.MAX_BUDGET, interval=(0, 50),  signals=self.signals)

    def run(self):
        result = staliro(self.model, self.specification, self.optimizer, self.options)