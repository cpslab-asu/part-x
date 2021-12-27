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
from ..executables.generate_statistics import generate_statistics
from ..models.results import Result
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import logging
import time
import datetime
import re, csv, itertools
import pathlib
import os

def run_partx(benchmark_name, test_function, test_function_dimension, region_support, 
              initialization_budget, maximum_budget, continued_sampling_budget, number_of_BO_samples, 
              NGP, M, R, branching_factor, nugget_mean, nugget_std_dev, alpha, delta,
              number_of_macro_replications, initial_seed, fv_quantiles_for_gp, results_at_confidence, gpr_params, results_folder_name):
    
    
    # create a directory for storing result files
    base_path = pathlib.Path()
    result_directory = base_path.joinpath(results_folder_name)
    result_directory.mkdir(exist_ok=True)
    benchmark_result_directory = result_directory.joinpath(benchmark_name)
    benchmark_result_directory.mkdir(exist_ok=True)
    benchmark_result_pickle_files = benchmark_result_directory.joinpath(benchmark_name + "_result_generating_files")
    benchmark_result_pickle_files.mkdir(exist_ok=True)

    results_csv = benchmark_result_directory.joinpath(benchmark_name + "_results_csv")
    results_csv.mkdir(exist_ok=True)
    
    # create partx options
    options = partx_options(region_support, branching_factor, test_function_dimension, number_of_BO_samples, alpha, M, R,  
                            delta, True, initialization_budget, maximum_budget, continued_sampling_budget, 
                            nugget_mean, nugget_std_dev, initial_seed, fv_quantiles_for_gp, benchmark_name, NGP, gpr_params)
    
    f = open(benchmark_result_pickle_files.joinpath(options.BENCHMARK_NAME + "_options.pkl"), "wb")
    pickle.dump(options,f)
    f.close()


    # Start running

    inputs = []

    for replication_number in range(number_of_macro_replications):
        data = [replication_number, options, test_function, benchmark_result_directory]
        inputs.append(data)
        results = run_single_replication(data)
    
        
    # print("Starting run for {} macro replications".format(len(inputs)))
    # pool = Pool()

    # with Pool(processes=os.cpu_count() - 1) as pool:
    #     for replication_number in range(number_of_macro_replications):
    #         result = pool.apply_async(run_single_replication, (inputs[replication_number],))
            # result_objs.append(result)
        
        # results = [result.get() for result in result_objs]
        # print(len(results), np.mean(results), np.var(results))

    # with Pool(10) as pool:
    #     results = list(pool.map(run_single_replication, inputs))

    # results = list(pool.map(run_single_replication, inputs))

    result_dictionary = generate_statistics(options.BENCHMARK_NAME, number_of_macro_replications, options.fv_quantiles_for_gp, results_at_confidence,results_folder_name)

    today = time.strftime("%m/%d/%Y")
    file_date = today.replace("/","_")
    values = []
    with open(results_csv.joinpath(options.BENCHMARK_NAME+"_"+file_date+ "_results.csv"), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in result_dictionary.items():
            writer.writerow([key, value])
            values.append(value)
    print("Done")
    result = Result(*values)

    return [result]