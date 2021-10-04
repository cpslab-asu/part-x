from ..numerical.classification import calculate_volume
from ..utilities.utils_partx import assign_budgets, branch_new_region_support, pointsInSubRegion, plotRegion

from ..models.partx_node import partx_node
from ..models.partx_options import partx_options
import numpy as np
from ..numerical.classification import calculate_volume
import matplotlib.pyplot as plt
from ..numerical.budget_check import budget_check
from treelib import Tree
from ..numerical.calIntegral import calculate_mc_integral
from ..numerical.utils_partx import plotRegion
from .single_replication import run_single_replication
from pathos.multiprocessing import ProcessingPool as Pool
from .exp_statistics import get_true_fv
import pickle
import logging


def test_function(X):  ##CHANGE
    return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 # Himmelblau's
    # return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) - 20 # Rosenbrock
    # return (1 + (X[0] + X[1] + 1) ** 2 * (
    #             19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
    #                    30 + (2 * X[0] - 3 * X[1]) ** 2 * (
    #                        18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50



BENCHMARK_NAME = "Himmelblaus_3"

base_path = pathlib.Path()

result_directory = base_path.joinpath('result_files')
result_directory.mkdir(exist_ok=True)

benchmark_result_directory = result_directory.joinpath(BENCHMARK_NAME)
benchmark_result_directory.mkdir(exist_ok=True)

benchmark_result_pickle_files = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_result_generating_files")
benchmark_result_pickle_files.mkdir(exist_ok=True)


# Options initialization

# Test function properties
test_function_dimension = 2
region_support = np.array([[[-5., 5.], [-5., 5.]]])

# Budgets
initialization_budget = 10
max_budget = 5000
continued_sampling_budget = 100

# BO grid size : number_of_BO_samples * number_of_samples_gen_GP
number_of_BO_samples = [20]
# R = number_of_BO_samples[0]
# M = number_of_samples_gen_GP
R = 20
M = 500
NGP = R * M

# Mostly not changes. change with caution
branching_factor = 2
nugget_mean = 0
nugget_std_dev = 0.001
alpha = [0.95]
delta = 0.001

# Other Parameters
number_of_macro_replications = 50
start_seed = 1000
fv_quantiles_for_gp = [0.5,0.95,0.99]

# Build options
options = partx_options(region_support, branching_factor, test_function_dimension, 
                        number_of_BO_samples, alpha, M, R, 
                        delta, True, initialization_budget, max_budget, 
                        continued_sampling_budget, nugget_mean, nugget_std_dev, start_seed, fv_quantiles_for_gp, BENCHMARK_NAME, NGP)

f = open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME + "_options.pkl"), "wb")
pickle.dump(options,f)
f.close()

### Uniform Monte Carlo

####################

inputs = []

for replication_number in range(number_of_macro_replications):

    data = [replication_number, options, test_function, benchmark_result_directory]
    inputs.append(data)
    
print(len(inputs))
pool = Pool()
results = list(pool.map(run_single_replication, inputs))

points_for_unif_sampling = 10000
rng = np.random.default_rng(10000)
true_fv, x,y = get_true_fv(points_for_unif_sampling, options, rng, test_function)
mc_uniform_test_function = {"true_fv" : true_fv,
                            "x" : x,
                            "y" : y}

f = open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME + "_mc_truefv_test_function.pkl"), "wb")
pickle.dump(mc_uniform_test_function, f)
f.close()