import numpy as np
from partx.interfaces.run_standalone import run_partx

def test_function(X):
    return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) - 20 # Rosenbrock
    

# Test function properties
test_function_dimension = 2
region_support = np.array([[[-1., 1.], [-1., 1.]]])

# Budgets
initialization_budget = 10
max_budget = 5000
continued_sampling_budget = 100
number_of_BO_samples = [10]

# Quantile Estimation Parameters
R = 20
M = 500

# Mostly not changes. change with caution
branching_factor = 2
nugget_mean = 0
nugget_std_dev = 0.001
alpha = [0.05]
delta = 0.001

# Other Parameters
number_of_macro_replications = 1
start_seed = 1000
fv_quantiles_for_gp = [0.01, 0.05, 0.5]


results_folder_name = "results"
BENCHMARK_NAME = "rosenbrock"
results_at_confidence = 0.95
gpr_params = list(["kriging",5])

num_cores = 2

results_at_confidence = 0.95
run_partx(BENCHMARK_NAME, test_function, test_function_dimension, region_support, 
              initialization_budget, max_budget, continued_sampling_budget, number_of_BO_samples, 
              M, R, branching_factor, alpha, delta,
              number_of_macro_replications, start_seed, fv_quantiles_for_gp, results_at_confidence, gpr_params, results_folder_name, num_cores)