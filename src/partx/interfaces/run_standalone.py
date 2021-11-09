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
from ..executables.single_replication import run_single_replication
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import logging

import pathlib


def run_partx(benchmark_name, test_function, test_function_dimension, region_support, 
              initialization_budget, maximum_budget, continued_sampling_budget, number_of_BO_samples, 
              NGP, M, R, branching_factor, nugget_mean, nugget_std_dev, alpha, delta,
              number_of_macro_replications, initial_seed, fv_quantiles_for_gp, results_folder_name):
    
    
    # create a directory for storing result files
    base_path = pathlib.Path()
    result_directory = base_path.joinpath(results_folder_name)
    result_directory.mkdir(exist_ok=True)
    benchmark_result_directory = result_directory.joinpath(benchmark_name)
    benchmark_result_directory.mkdir(exist_ok=True)
    benchmark_result_pickle_files = benchmark_result_directory.joinpath(benchmark_name + "_result_generating_files")
    benchmark_result_pickle_files.mkdir(exist_ok=True)

    
    # create partx options
    options = partx_options(region_support, branching_factor, test_function_dimension, number_of_BO_samples, alpha, M, R,  
                            delta, True, initialization_budget, maximum_budget, continued_sampling_budget, 
                            nugget_mean, nugget_std_dev, initial_seed, fv_quantiles_for_gp, benchmark_name, NGP)
    
    f = open(benchmark_result_pickle_files.joinpath(options.BENCHMARK_NAME + "_options.pkl"), "wb")
    pickle.dump(options,f)
    f.close()


    # Start running

    inputs = []

    for replication_number in range(number_of_macro_replications):
        data = [replication_number, options, test_function, benchmark_result_directory]
        inputs.append(data)
         
    print("Starting run for {} macro replications".format(len(inputs)))
    pool = Pool()
    results = list(pool.map(run_single_replication, inputs))
    
    return [results]