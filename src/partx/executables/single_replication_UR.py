from ..numerical.classification import calculate_volume
from ..utilities.utils_partx import assign_budgets, branch_new_region_support, pointsInSubRegion, plotRegion
from ..models.testFunction import callCounter
from ..models.partx_node import partx_node
from ..models.partx_options import partx_options
import numpy as np
from ..numerical.classification import calculate_volume
import matplotlib.pyplot as plt
from ..numerical.budget_check import budget_check
from treelib import Tree
from ..numerical.calIntegral import calculate_mc_integral
import logging
import pickle
from .exp_statistics import falsification_volume_using_gp
from ..numerical.sampling import uniform_sampling
from ..numerical.calculate_robustness import calculate_robustness

def run_single_replication_UR(inputs):
    replication_number, options, test_function, benchmark_result_directory = inputs

    print("Started Replication Number {} with {} points.".format(replication_number, options.number_of_samples))
    seed = options.start_seed + replication_number
    BENCHMARK_NAME = options.BENCHMARK_NAME

    benchmark_result_pickle_files = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_result_generating_files")
    benchmark_result_pickle_files.mkdir(exist_ok=True)

    callCounts = callCounter(test_function)
    rng = np.random.default_rng(seed)

    samples = uniform_sampling(options.number_of_samples, options.initial_region_support, options.test_function_dimension, rng)
    y = calculate_robustness(samples, callCounts)

    true_fv = (np.sum(np.array(y <= 0)) / (options.number_of_samples)) * calculate_volume(options.initial_region_support)
    result_dictionary = {"true_fv" : true_fv,
                         "samples" : samples,
                         "robustness" : y}

    print("Ended Replication Number {} with {} points.".format(replication_number, options.number_of_samples))

    f = open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME + "_" + str(replication_number) + "_uniform_random_results.pkl"), "wb")
    pickle.dump(result_dictionary, f)
    f.close()

    return {
        'result_dictionary': result_dictionary
    }