import pickle
import pathlib
import os
import time
import csv
import numpy as np
from typing import Callable, Dict, Tuple, Union
from numpy.typing import NDArray
from pathos.multiprocessing import ProcessingPool as Pool
from dataclasses import dataclass
from typing import Any, List, Sequence, Callable
from treelib import Tree
from staliro import Sample
from staliro.optimizers import ObjFunc, Optimizer

from .gpr import GPRSkeleton
from .bo import BO_Interface
from .optimizer import PartXOptions, run_single_replication
from .results import generate_statistics
from .utils import OracleCreator

def run_partx(BENCHMARK_NAME:str, test_function:Callable[[NDArray], float], oracle_function, num_macro_reps:int, init_reg_sup:NDArray, tf_dim:int,
                max_budget:int, init_budget:int, bo_budget:int, cs_budget:int, 
                n_tries_randomsampling:int, n_tries_BO:int, 
                alpha:float, R:int, M:int, delta:float, fv_quantiles_for_gp:list,
                branching_factor:int, uniform_partitioning:bool, start_seed:int, 
                gpr_model:GPRSkeleton, bo_model:BO_Interface, 
                init_sampling_type:str, cs_sampling_type:str, 
                q_estim_sampling:str, mc_integral_sampling_type:str, 
                results_sampling_type:str, 
                results_at_confidence:float, results_folder_name:str, num_cores:int):
    
    """_summary_

    Args:
            BENCHMARK_NAME: Benchmark Name for book-keeping purposes
            test_function: Test Function meant for Optimization
            num_macro_rep: Number of Macro-Replications intended to be done
            init_reg_sup: Initial Region Support. Expcted a 2d numpy array of Nx2. N is the number of dimensions, colum1 refers to lower bounds and column 2 refers to upper bounds.
            tf_dim: Dimesnionality of the problem
            max_budget : Maximum Budget for which Part-X should run
            init_budget: Initial Sampling Budget for any subregion
            bo_budget: Bayesian Optimization Samples budget for evey subregion
            cs_budget: Continued Sampling Budget for Classified Regions
            alpha: Region Classification Percentile
            R: The number of monte-carlo iterations. This is used for calculation of quantiles of a region.
            M: The number of evaluations per monte-carlo iteration. This is used for calculation of quantiles of a region.
            delta: A number used to define the fraction of dimension, below which no further brnching in that dimension takes place. It is used for clsssificastion of a region.
            fv_quantiles_for_gp: List of values used for calculation at certain quantile values.
            branching_factor: Number of sub-regions in which a region is branched.
            uniform_partitioning: Wether to perform Uniform Partitioning or not. 
            start_seed: Starting Seed of Experiments
            gpr_model: GPR Model bas on the GPR interface.
            bo_model: Bayesian Optimization Model based on BO interface.
            init_sampling_type: Initial Sampling Algorithms. Defaults to "lhs_sampling".
            cs_sampling_type: Continued Sampling Mechanism. Defaults to "lhs_sampling".
            q_estim_sampling: Quantile estimation sampling Mechanism. Defaults to "lhs_sampling".
            mc_integral_sampling_type: Monte Carlo Integral Sampling Mechanism. Defaults to "lhs_sampling".
            results_sampling_type: Results Sampling Mechanism. Defaults to "lhs_sampling".
            results_at_confidence: Confidence level at which result to be computed
            results_folder_name: Results folder name, 
            num_cores: Num cores over which the it can be parallelized
        
    Raises:
        Exception: If num_cores over the available cores

    Returns:
        dict: dictionary of results
    """
    # create a directory for storing result files
    
    base_path = pathlib.Path()
    result_directory = base_path.joinpath(results_folder_name)
    result_directory.mkdir(exist_ok=True)
    benchmark_result_directory = result_directory.joinpath(BENCHMARK_NAME)
    benchmark_result_directory.mkdir(exist_ok=True)
    
    benchmark_result_pickle_files = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_result_generating_files")
    benchmark_result_pickle_files.mkdir(exist_ok=True)

    results_csv = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_results_csv")
    results_csv.mkdir(exist_ok=True)
    
    
    # create partx options
    options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, 
                init_sampling_type, cs_sampling_type, 
                q_estim_sampling, mc_integral_sampling_type, 
                results_sampling_type)
    
    oracle_info = OracleCreator(oracle_function, n_tries_randomsampling, n_tries_BO)
    
    with open(benchmark_result_pickle_files.joinpath(options.BENCHMARK_NAME + "_options.pkl"), "wb") as f:
        pickle.dump(options,f)
    

    # Start running

    inputs = []
    if num_cores == 0:
        raise Exception("Number of cores to use cannot be 0")
    elif num_cores == 1:
        print("Running without parallalization")
        results = []
        for replication_number in range(num_macro_reps):
            data = (replication_number, options, test_function, oracle_info, benchmark_result_directory)
            inputs.append(data)
            res = run_single_replication(data)
            results.append(res)
    elif num_cores != 1:
        num_cores_available = min((os.cpu_count() - 1), num_cores) # type: ignore
        if num_cores == num_cores_available:
            print("Running with {} cores".format(num_cores_available))
        elif num_cores > num_cores_available:
            print("Cannot run with {} cores. Instead running with {} cores.".format(num_cores, num_cores_available))
        elif num_cores < num_cores_available:
            print("Max cores uitilised can be {}. Instead running with {} cores.".format((os.cpu_count() - 1), num_cores_available)) # type: ignore
        for replication_number in range(num_macro_reps):
            data = (replication_number, options,  test_function, oracle_info, benchmark_result_directory)
            inputs.append(data)
        with Pool(num_cores_available) as pool:
            results = list(pool.map(run_single_replication, inputs))
        pool.close()
    result_dictionary = generate_statistics(options.BENCHMARK_NAME, num_macro_reps, options.fv_quantiles_for_gp, results_at_confidence,results_folder_name)
    
    today = time.strftime("%m/%d/%Y")
    file_date = today.replace("/","_")
    values = []
    with open(results_csv.joinpath(options.BENCHMARK_NAME+"_"+file_date+ "_results.csv"), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in result_dictionary.items():
            writer.writerow([key, value])
            values.append(value)
    print("Done")
    # result = Result(*values)

    return results



PartXResult = list[Tuple[Tree, Dict[str, Union[float, NDArray]], NDArray, List]]

@dataclass(frozen=True)
class PartX(Optimizer[float, PartXResult]):
    """The PartX optimizer provides statistical guarantees about the existence of falsifying behaviour in a system."""

    BENCHMARK_NAME: str
    oracle_function: Callable[[NDArray], float]|None
    num_macro_reps: int
    init_budget: int
    bo_budget: int
    cs_budget: int
    n_tries_randomsampling: int
    n_tries_BO: int
    alpha: float
    R: int
    M: int
    delta: float
    fv_quantiles_for_gp: List[float]
    branching_factor: int
    uniform_partitioning: bool
    seed: int
    gpr_model: GPRSkeleton
    bo_model: BO_Interface
    init_sampling_type: str
    cs_sampling_type: str
    q_estim_sampling: str
    mc_integral_sampling_type: str
    results_sampling_type: str
    results_at_confidence: float
    results_folder_name: str
    num_cores: int

    # def __init__(self,
    #                 BENCHMARK_NAME: str,
    #                 oracle_function: Callable[[NDArray], float],
    #                 num_macro_reps: int,
    #                 init_budget: int,
    #                 bo_budget: int,
    #                 cs_budget: int,
    #                 n_tries_randomsampling: int,
    #                 n_tries_BO: int,
    #                 alpha: float,
    #                 R: int,
    #                 M: int,
    #                 delta: float,
    #                 fv_quantiles_for_gp: List[float],
    #                 branching_factor: int,
    #                 uniform_partitioning: bool,
    #                 seed: int,
    #                 gpr_model: GPRSkeleton,
    #                 bo_model: BO_Interface,
    #                 init_sampling_type: str,
    #                 cs_sampling_type: str,
    #                 q_estim_sampling: str,
    #                 mc_integral_sampling_type: str,
    #                 results_sampling_type: str,
    #                 results_at_confidence: float,
    #                 results_folder_name: str,
    #                 num_cores: int
    #     ):
        
    #     self.BENCHMARK_NAME = BENCHMARK_NAME
    #     self.oracle_function = oracle_function
    #     self.num_macro_reps = num_macro_reps
    #     self.init_budget = init_budget
    #     self.bo_budget = bo_budget
    #     self.cs_budget = cs_budget
    #     self.n_tries_randomsampling = n_tries_randomsampling
    #     self.n_tries_BO = n_tries_BO
    #     self.alpha = alpha
    #     self.R = R
    #     self.M = M
    #     self.delta = delta
    #     self.fv_quantiles_for_gp = fv_quantiles_for_gp
    #     self.branching_factor = branching_factor
    #     self.uniform_partitioning = uniform_partitioning
    #     self.seed = seed
    #     self.gpr_model = gpr_model
    #     self.bo_model = bo_model
    #     self.init_sampling_type = init_sampling_type
    #     self.cs_sampling_type = cs_sampling_type
    #     self.q_estim_sampling = q_estim_sampling
    #     self.mc_integral_sampling_type = mc_integral_sampling_type
    #     self.results_sampling_type = results_sampling_type
    #     self.results_at_confidence = results_at_confidence
    #     self.results_folder_name = results_folder_name
    #     self.num_cores = num_cores

    def optimize(self, func: ObjFunc[float], params: Optimizer.Params) -> PartXResult:
        region_support = np.array(params.input_bounds)

        print("************************************************************************")
        print("************************************************************************")
        print("************************************************************************")
        print(f"Test Function:\n Testing function is a {region_support.shape[0]}d problem with initial region support of {region_support}.")
        print(f"Starting {self.num_macro_reps} macro replications with maximum budget of {params.budget}, where")
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
        
        
        # def test_function(sample: NDArray) -> float:
        #     return func.eval_sample(Sample(sample))
    
        
        return PartXResult(
            run_partx(
                BENCHMARK_NAME = self.BENCHMARK_NAME,
                test_function = func.eval_sample,
                oracle_function= self.oracle_function,
                num_macro_reps = self.num_macro_reps,
                init_reg_sup = region_support,
                tf_dim = region_support.shape[0],
                max_budget = params.budget,
                init_budget = self.init_budget,
                bo_budget = self.bo_budget,
                cs_budget = self.cs_budget,
                n_tries_randomsampling= self.n_tries_randomsampling,
                n_tries_BO=self.n_tries_BO,
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
        )
    