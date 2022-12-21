
from ..utils import calculate_volume
import numpy as np
from numpy.typing import NDArray
from typing import Callable


class PartXOptions:
    def __init__(self, BENCHMARK_NAME:str, init_reg_sup:NDArray, tf_dim: int,
                max_budget: int, init_budget:int, bo_budget:int, cs_budget:int, 
                alpha:float, R:int, M:int, delta:float, fv_quantiles_for_gp:list,
                branching_factor:int, uniform_partitioning:bool, start_seed:int, 
                gpr_model:Callable, bo_model:Callable, 
                init_sampling_type:str = "lhs_sampling", cs_sampling_type:str = "lhs_sampling", 
                q_estim_sampling:str = "lhs_sampling", mc_integral_sampling_type:str = "lhs_sampling", 
                results_sampling_type:str = "lhs_sampling"):

        """Helps to set up options for Part-X

        Args:
            BENCHMARK_NAME: Benchmark Name for book-keeping purposes
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
        """
        self.BENCHMARK_NAME = BENCHMARK_NAME   

        self.init_reg_sup = init_reg_sup.astype('float64')
        self.tf_dim = tf_dim

        self.max_budget = max_budget
        self.init_budget = init_budget
        self.bo_budget = bo_budget
        self.cs_budget = cs_budget

        self.init_sampling_type = init_sampling_type
        self.cs_sampling_type = cs_sampling_type
        self.q_estim_sampling = q_estim_sampling
        self.mc_integral_sampling_type = mc_integral_sampling_type
        self.results_sampling_type = results_sampling_type

        self.alpha = alpha
        self.R = R
        self.M = M
        self.delta = delta
        self.fv_quantiles_for_gp = fv_quantiles_for_gp
        self.min_volume = (self.delta ** self.tf_dim) * calculate_volume(self.init_reg_sup)

        self.branching_factor = branching_factor
        self.uniform_partitioning = uniform_partitioning
        self.start_seed = start_seed
        
        self.gpr_model = gpr_model
        self.bo_model = bo_model
