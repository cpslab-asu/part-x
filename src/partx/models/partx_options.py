from ..numerical.classification import calculate_volume
import numpy as np

class partx_options:
    def __init__(self, initial_region_support, branching_factor, test_function_dimension,
                number_of_BO_samples, alpha, M, R, 
                delta, uniform_partitioning, initialization_budget, max_budget, 
                continued_sampling_budget, nugget_mean, nugget_std_dev, start_seed, fv_quantiles_for_gp, BENCHMARK_NAME, NGP):
                
        self.initial_region_support = initial_region_support.astype('float64')
        self.test_function_dimension = test_function_dimension
        # self.number_of_samples_gen_GP = number_of_samples_gen_GP
        self.alpha = alpha
        self.M = M
        self.R = R
        self.delta = delta
        self.uniform_partitioning = uniform_partitioning
        self.min_volume = (self.delta ** self.test_function_dimension) * calculate_volume(self.initial_region_support)[0]
        self.initialization_budget = initialization_budget
        self.max_budget = max_budget
        self.number_of_BO_samples = number_of_BO_samples
        self.continued_sampling_budget = continued_sampling_budget
        self.branching_factor = branching_factor
        self.nugget_mean = nugget_mean
        self.nugget_std_dev = nugget_std_dev
        self.start_seed = start_seed
        self.fv_quantiles_for_gp = fv_quantiles_for_gp
        self.BENCHMARK_NAME = BENCHMARK_NAME
        self.NGP = NGP
