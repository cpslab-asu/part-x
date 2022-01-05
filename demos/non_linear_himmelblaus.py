import numpy as np
from partx.interfaces.run_standalone import run_partx

def test_function(X):
    return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 # Himmelblau's

# Test function properties
test_function_dimension = 2
region_support = np.array([[[-5., 5.], [-5., 5.]]])

# Budgets
initialization_budget = 10
max_budget = 5000
continued_sampling_budget = 100

# BO grid size : number_of_BO_samples * number_of_samples_gen_GP
number_of_BO_samples = [10]
R = 20
M = 500
NGP = R*M

# Mostly not changes. change with caution
branching_factor = 2
nugget_mean = 0
nugget_std_dev = 0.001
alpha = [0.05]
delta = 0.001

# Other Parameters
number_of_macro_replications = 50
start_seed = 1000
fv_quantiles_for_gp = [0.01, 0.05, 0.5]


results_at_confidence = 0.95
results_folder_name = "results"
BENCHMARK_NAME = "himmelblaus"
results_at_confidence = 0.95
gpr_params = 5

num_cores = 2

results_at_confidence = 0.95
run_partx(BENCHMARK_NAME, test_function, test_function_dimension, region_support, 
              initialization_budget, max_budget, continued_sampling_budget, number_of_BO_samples, 
              NGP, M, R, branching_factor, alpha, delta,
              number_of_macro_replications, start_seed, fv_quantiles_for_gp, results_at_confidence, gpr_params, results_folder_name, num_cores)