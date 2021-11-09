from ..models.uniform_random_options import uniform_random_options
from ..executables.single_replication_UR import run_single_replication_UR
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import pathlib


def run_partx_UR(number_of_macro_replications, benchmark_name, initial_seed, test_function, 
                test_function_dimension, region_support,
                number_of_samples, results_folder):
    
    
    # create a directory for storing result files
    base_path = pathlib.Path()
    result_directory = base_path.joinpath(results_folder)
    result_directory.mkdir(exist_ok=True)
    benchmark_result_directory = result_directory.joinpath(benchmark_name)
    benchmark_result_directory.mkdir(exist_ok=True)
    benchmark_result_pickle_files = benchmark_result_directory.joinpath(benchmark_name + "_result_generating_files")
    benchmark_result_pickle_files.mkdir(exist_ok=True)

    
    # create partx options
    options = uniform_random_options(initial_seed, number_of_samples, region_support, test_function_dimension, benchmark_name)
    
    f = open(benchmark_result_pickle_files.joinpath(options.BENCHMARK_NAME + "_uniform_random_options.pkl"), "wb")
    pickle.dump(options,f)
    f.close()


    # Start running

    inputs = []

    for replication_number in range(number_of_macro_replications):
        data = [replication_number, options, test_function, benchmark_result_directory]
        inputs.append(data)
        run_single_replication_UR(data)
    
        
    # print("Starting run for {} macro replications".format(len(inputs)))
    # pool = Pool()
    # results = list(pool.map(run_single_replication_UR, inputs))
    results = 1
    return results