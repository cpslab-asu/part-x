import logging
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from treelib import Tree
from treelib.node import Node

from .bo import BO_Interface, BOSampling
from .gpr import GPRSkeleton
from .results import fv_using_gp
from .sampling import OOBError, lhs_sampling, uniform_sampling
from .stats import assign_budgets, calculate_mc_integral, classification, estimate_quantiles
from .utils import (
    Fn,
    OracleCreator,
    branch_region,
    calculate_volume,
    compute_robustness,
    divide_points,
)


class PartXOptions:
    def __init__(self, 
                 BENCHMARK_NAME:str, 
                 init_reg_sup:NDArray[np.float_], 
                 tf_dim: int,
                 max_budget: int, 
                 init_budget:int, 
                 bo_budget:int, 
                 cs_budget:int, 
                 alpha:float, 
                 R:int, 
                 M:int, 
                 delta:float, 
                 fv_quantiles_for_gp:list[float], 
                 branching_factor:int, 
                 uniform_partitioning:bool, 
                 start_seed:int, 
                 gpr_model:GPRSkeleton, 
                 bo_model:BO_Interface, 
                 init_sampling_type:str = "lhs_sampling", 
                 cs_sampling_type:str = "lhs_sampling", 
                 q_estim_sampling:str = "lhs_sampling",
                 mc_integral_sampling_type:str = "lhs_sampling", 
                 results_sampling_type:str = "lhs_sampling" ) -> None:

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


class PartXNode:
    def __init__(self, self_id: int, parent_id: int, region_support: NDArray, samples_in: NDArray, samples_out: NDArray, branch_dir:int, region_class: str = 'r'):
        """PartXNode Operations for various types of region classifications.

        Args:
            self_id: Identifier of Node to allow identification of bugs after tree is created.
            parent_id: Identifier of Parent Node to allow identification of bugs after tree is created.
            region_support: Min and Max of all dimensions
            samples_in: Samples from Training set.
            samples_out: Evaluated values of samples from Training set.
            branch_dir: Direction in which this node should be brached (if needed).
            region_class (str, optional): Current Type of region. Defaults to 'r'.
        """

        self.self_id = self_id
        self.parent_id = parent_id
        self.region_support = region_support
        self.region_class = region_class
        self.samples_in = samples_in
        self.samples_out = samples_out
        self.branch_dir = branch_dir
        

    def samples_management_unclassified(self, test_function: Fn, options: PartXOptions, oracle_info: OracleCreator, rng: Generator) -> str:
        """Method to manage samples in subregion which is unclassified (r, r+, r-)

        Args:
            test_function: Function of System Under Test.
            options: PartX Options object
            rng: RNG object from numpy

        Raises:
            ValueError: If options.init_sampling_type is not defined correctly.

        Returns:
            Class of new region
        """
        
        assert self.region_class == "r" or self.region_class == "r+" or self.region_class == "r-"
        samples_present = self.samples_out.shape[0]
        init_sampling_left = options.init_budget - samples_present
        
        try:
            if init_sampling_left > 0:
                if options.init_sampling_type == "lhs_sampling":
                    x_init_extra = lhs_sampling(init_sampling_left, self.region_support, options.tf_dim, oracle_info, rng)
                elif options.init_sampling_type == "uniform_sampling":
                    x_init_extra = uniform_sampling(init_sampling_left, self.region_support, options.tf_dim, oracle_info, rng)
                else:
                    raise ValueError(f"{options.init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
                
                y_init_extra = compute_robustness(x_init_extra, test_function)
                if self.samples_out.shape[0] == 0:
                    new_samples_in = x_init_extra
                    new_samples_out = y_init_extra
                else:
                    new_samples_in = np.concatenate((self.samples_in,x_init_extra), axis = 0)
                    new_samples_out = np.concatenate((self.samples_out,y_init_extra), axis = 0)
            else:
                new_samples_in = self.samples_in
                new_samples_out = self.samples_out

            bo = BOSampling(options.bo_model)
            final_new_samples_in, final_new_samples_out, _, _ = bo.sample(test_function, options.bo_budget, new_samples_in, new_samples_out, self.region_support, options.gpr_model, oracle_info, rng)
            self.samples_in = final_new_samples_in
            self.samples_out = final_new_samples_out

            self.lower_bound, self.upper_bound = estimate_quantiles(self.samples_in, self.samples_out, self.region_support, options.tf_dim, options.alpha,options.R,options.M, options.gpr_model, oracle_info, rng, options.q_estim_sampling)
            
            self.new_region_class = classification(self.region_support, self.region_class, options.min_volume, self.lower_bound, self.upper_bound)
            self.region_class = self.new_region_class
        except OOBError:
            self.new_region_class = "i"
            self.region_class = "i"
        return self.new_region_class
    
    def samples_management_classified(self, num_samples: int, test_function: Fn, options: PartXOptions, oracle_info: OracleCreator, rng: Generator, fin_cs: bool = False) -> str:
        """Method to manage samples in where continued sampling is to be performed.

        Args:
            num_samples: Number oimport numpy as np
            test_function: Function of System Under Test.
            options: PartX Options object
            rng: RNG object from numpy
            fin_cs: If False, performing continued sampling requires regions class be classified (+, -),  Defaults to False.

        Raises:
            ValueError: options.cs_sampling_type not correctly defined

        Returns:
            Class of new region
        """
        if not fin_cs:
            assert self.region_class == "+" or self.region_class == "-"
        else:
            assert self.region_class == "r" or self.region_class == "r+" or self.region_class == "r-" or self.region_class == "+" or self.region_class == "-"
        
        if options.cs_sampling_type == "lhs_sampling":
            cs_samples_in = lhs_sampling(num_samples, self.region_support, options.tf_dim, oracle_info, rng)
        elif options.cs_sampling_type == "uniform_sampling":
            cs_samples_in = uniform_sampling(num_samples, self.region_support, options.tf_dim, oracle_info, rng)
        else:
            raise ValueError(f"{options.cs_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")


        cs_samples_out = compute_robustness(cs_samples_in, test_function)
        self.samples_in = np.concatenate((self.samples_in, cs_samples_in), axis=0)
        self.samples_out = np.concatenate((self.samples_out, cs_samples_out), axis=0)
        
        self.lower_bound, self.upper_bound = estimate_quantiles(self.samples_in, self.samples_out, self.region_support, options.tf_dim, options.alpha,options.R,options.M, options.gpr_model, oracle_info, rng)
        
        self.new_region_class = classification(self.region_support, self.region_class, options.min_volume, self.lower_bound, self.upper_bound)
        self.region_class = self.new_region_class

        return self.new_region_class

def run_single_replication(inputs: tuple[int, PartXOptions, Callable[[NDArray[np.float_]], float], OracleCreator, Path])->tuple[Any, Any,Any, Any]:
    
    replication_number, options, test_function, oracle_info, benchmark_result_directory = inputs

    seed = options.start_seed + replication_number
    BENCHMARK_NAME = options.BENCHMARK_NAME
    
    benchmark_result_log_files = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_log_files")

    benchmark_result_log_files.mkdir(exist_ok=True)

    benchmark_result_pickle_files = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_result_generating_files")
    benchmark_result_pickle_files.mkdir(exist_ok=True)

    tf_wrapper = Fn(test_function)
    
    log = logging.getLogger()
    log.setLevel(logging.INFO) 
    fh = logging.FileHandler(filename=benchmark_result_log_files.joinpath(BENCHMARK_NAME + "_" + str(replication_number) + ".log"))
    formatter = logging.Formatter(
                    fmt = '%(asctime)s :: %(message)s', datefmt = '%a, %d %b %Y %H:%M:%S'
                    )

    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.info(f"Information about Replication {replication_number}")
    log.info(f"Running {BENCHMARK_NAME} Replication {replication_number} with seed {seed}")
    log.info("**************************************************")
    log.info("Options File:")
    options_results = vars(options)
    for key, value in options_results.items():
        log.info(f"{key} : {value}")
    log.info("**************************************************")
    log.info(f"Budget Used = {tf_wrapper.count}")
    log.info(f"Budget Available (Max Budget) = {options.max_budget}")
    log.info("**************************************************")
    log.info("**************************************************")
    log.info("***************Replication Start******************")
    print(f"Started replication {replication_number}")


    rng = np.random.default_rng(seed)

    if options.max_budget < options.init_budget + options.bo_budget:
        log.info("Error: Cannot Initialize root node")
        raise Exception("(Max Budget) MUST NOT BE LESS THAN (Initialization_budget + number_of_BO_samples)")

    samples_in = np.array([[]])
    samples_out = np.array([])
    branch_dir_order = np.arange(options.tf_dim)
    direction = rng.permutation(branch_dir_order)
    direction_count = 0

    total_time_start = time.perf_counter()

    remaining_regions_l = []
    classified_region_l = []
    unidentified_regions_l = []
    infeasible_regions_l = []

    node_id_keeper = 0
    root = PartXNode(node_id_keeper, node_id_keeper, options.init_reg_sup, samples_in, samples_out, direction_count, region_class="r")
    root.samples_management_unclassified(tf_wrapper, options, oracle_info, rng)
    ftree = Tree()
    ftree.create_node(node_id_keeper, node_id_keeper, data = root)

    if root.region_class == 'r' or root.region_class == 'r+' or root.region_class == 'r-':
        remaining_regions_l.append(node_id_keeper)
    elif root.region_class == '+' or root.region_class == '-':
        classified_region_l.append(node_id_keeper)
    elif root.region_class == 'u':
        unidentified_regions_l.append(node_id_keeper)
    elif root.region_class == "i":
        infeasible_regions_l.append(node_id_keeper)
    
    log.info("**************************************************")
    log.info(f"Remaining Regions: {remaining_regions_l}")
    log.info(f"Classified Regions: {classified_region_l}")
    log.info(f"Unidentified Regions: {unidentified_regions_l}")
    log.info(f"Infeasible Regions: {infeasible_regions_l}")
    log.info(f"{tf_wrapper.count} Evaluations completed")
    log.info(f"{options.max_budget - tf_wrapper.count} left")
    log.info("**************************************************")

    while (options.max_budget - tf_wrapper.count) > 0 and (remaining_regions_l or classified_region_l):
        temp_remaining_regions_l = []
        
        if remaining_regions_l:
            budget_for_branching = 0
            potential_children = []
            for node_id in remaining_regions_l:
                node:Node = ftree.get_node(node_id) # type: ignore
                node_identifier = node.identifier
                node_data:PartXNode = node.data # type: ignore
                sub_bounds = branch_region(node_data.region_support, direction[node_data.branch_dir%options.tf_dim], options.uniform_partitioning, options.branching_factor, rng)
                x_samples_divided, y_samples_divided = divide_points(node_data.samples_in, node_data.samples_out, sub_bounds)
                
                for branches in range(options.branching_factor): 
                    node_id_keeper += 1

                    child_reg_sup = sub_bounds[branches]
                    child_reg_samples_in = x_samples_divided[branches]
                    child_reg_samples_out = y_samples_divided[branches]
                    budget_for_branching += max(options.init_budget - child_reg_samples_in.shape[0], 0) + options.bo_budget
                    child_node = deepcopy(PartXNode(node_id_keeper, node_identifier, child_reg_sup, child_reg_samples_in, child_reg_samples_out, node_data.branch_dir+1, region_class="r"))  # type: ignore
                    potential_children.append(child_node)
        else:
            budget_for_branching = 0


        if budget_for_branching <= (options.max_budget - tf_wrapper.count) and remaining_regions_l and potential_children:
            temp_remaining_regions_l = []
            while potential_children:
                curr_node = potential_children.pop()
                self_id = curr_node.self_id
                parent_id = curr_node.parent_id
                
                curr_node.samples_management_unclassified(tf_wrapper, options, oracle_info, rng)
                ftree.create_node(self_id, self_id, parent = parent_id, data = curr_node)
                if curr_node.region_class == 'r' or curr_node.region_class == 'r+' or curr_node.region_class == 'r-':
                    temp_remaining_regions_l.append(self_id)
                elif curr_node.region_class == '+' or curr_node.region_class == '-':
                    classified_region_l.append(self_id)
                elif curr_node.region_class == 'u':
                    unidentified_regions_l.append(self_id)
                elif curr_node.region_class == 'i':
                    infeasible_regions_l.append(self_id)
                    
                
                
            remaining_regions_l = temp_remaining_regions_l
            temp_classified_region_l = []

            if classified_region_l:
                volumes = []
                cs_budget_allocated = min(options.cs_budget, (options.max_budget - tf_wrapper.count))
                for classi_node in classified_region_l:
                    node = ftree.get_node(classi_node) # type: ignore
                    node_data = node.data # type: ignore
                    cs_indicator = calculate_mc_integral(node_data.samples_in, node_data.samples_out, node_data.region_support, options.tf_dim, options.R, options.M, options.gpr_model, oracle_info, rng, sampling_type=options.mc_integral_sampling_type)
                    volumes.append(cs_indicator)
                
                if np.sum(volumes) != 0.0:
                    volume_distribution = volumes/np.sum(volumes)
                else:
                    volume_distribution = volumes

                # print(volume_distribution)
                
                assigned_budgets = assign_budgets(volume_distribution, cs_budget_allocated)

                for iterate, classi_node in enumerate(classified_region_l):
                    
                    if assigned_budgets[iterate] != 0:
                        node = ftree.get_node(classi_node) # type: ignore
                        node_identifier = node.identifier
                        node_data = node.data # type: ignore

                        node_data.samples_management_classified(assigned_budgets[iterate], tf_wrapper, options, oracle_info, rng)
                        ftree.update_node(node_identifier, tag = node_identifier, data = node_data)
                        if node_data.region_class == 'r' or node_data.region_class == 'r+' or node_data.region_class == 'r-':
                            remaining_regions_l.append(node_identifier)
                        elif node_data.region_class == '+' or node_data.region_class == '-':
                            temp_classified_region_l.append(node_identifier)
                        elif node_data.region_class == 'u':
                            unidentified_regions_l.append(node_identifier)
                        elif node_data.region_class == "i":
                            infeasible_regions_l.append(node_identifier)
                    else:
                        node = ftree.get_node(classi_node) # type: ignore
                        node_identifier = node.identifier
                        node_data = node.data # type: ignore
                        ftree.update_node(node_identifier, tag = node_identifier, data = node_data)

                        if node_data.region_class == 'r' or node_data.region_class == 'r+' or node_data.region_class == 'r-':
                            remaining_regions_l.append(node_identifier)
                        elif node_data.region_class == '+' or node_data.region_class == '-':
                            temp_classified_region_l.append(node_identifier)
                        elif node_data.region_class == 'u':
                            unidentified_regions_l.append(node_identifier)
                        elif node_data.region_class == "i":
                            infeasible_regions_l.append(node_identifier)
            classified_region_l = temp_classified_region_l
        elif (options.max_budget - tf_wrapper.count > 0):
            budget_left = options.max_budget - tf_wrapper.count
            all_regions = remaining_regions_l + classified_region_l
            temp_remaining_regions_l = []
            temp_classified_region_l = []
            log.info("**************************************************")
            log.info(f"Entering Last Phase with {options.max_budget - tf_wrapper.count} left")
            log.info(f"Remaining Regions: {remaining_regions_l}")
            log.info(f"Classified Regions: {classified_region_l}")
            log.info(f"Unidentified Regions: {unidentified_regions_l}")
            log.info(f"Infeasible Regions: {infeasible_regions_l}")
            log.info("**************************************************")
            if all_regions:
                volumes = []
                for all_nodes in all_regions:
                    node = ftree.get_node(all_nodes) # type: ignore
                    node_data = node.data # type: ignore
                    cs_indicator = calculate_volume(node_data.region_support)
                    volumes.append(cs_indicator)
                
                if np.sum(volumes) != 0.0:
                    volume_distribution = volumes/np.sum(volumes)
                else:
                    volume_distribution = volumes
                
                assigned_budgets = assign_budgets(volume_distribution, budget_left)

                for iterate, all_nodes in enumerate(all_regions):
                    
                    if assigned_budgets[iterate] != 0:
                        node = ftree.get_node(all_nodes) # type: ignore
                        node_identifier = node.identifier
                        node_data = node.data # type: ignore

                        node_data.samples_management_classified(assigned_budgets[iterate], tf_wrapper, options, oracle_info, rng, fin_cs = True)
                        ftree.update_node(node_identifier, tag = node_identifier, data = node_data)
                        if node_data.region_class == 'r' or node_data.region_class == 'r+' or node_data.region_class == 'r-':
                            temp_remaining_regions_l.append(node_identifier)
                        elif node_data.region_class == '+' or node_data.region_class == '-':
                            temp_classified_region_l.append(node_identifier)
                        elif node_data.region_class == 'u':
                            unidentified_regions_l.append(node_identifier)
                        elif node_data.region_class == "i":
                            infeasible_regions_l.append(node_identifier)
                    else:
                        node = ftree.get_node(all_nodes) # type: ignore
                        node_identifier = node.identifier
                        node_data = node.data # type: ignore
                        ftree.update_node(node_identifier, tag = node_identifier, data = node_data)

                        if node_data.region_class == 'r' or node_data.region_class == 'r+' or node_data.region_class == 'r-':
                            temp_remaining_regions_l.append(node_identifier)
                        elif node_data.region_class == '+' or node_data.region_class == '-':
                            temp_classified_region_l.append(node_identifier)
                        elif node_data.region_class == 'u':
                            unidentified_regions_l.append(node_identifier)
                        elif node_data.region_class == "i":
                            infeasible_regions_l.append(node_identifier)

            classified_region_l =  temp_classified_region_l
            remaining_regions_l = temp_remaining_regions_l
        log.info("**************************************************")
        log.info(f"Remaining Regions: {remaining_regions_l}")
        log.info(f"Classified Regions: {classified_region_l}")
        log.info(f"Unidentified Regions: {unidentified_regions_l}")
        log.info(f"Infeasible Regions: {infeasible_regions_l}")
        log.info(f"{tf_wrapper.count} Evaluations completed")
        log.info(f"{options.max_budget - tf_wrapper.count} left")
        log.info("**************************************************")
    
    log.info(f"**************************************************")
    log.info(f"*********Replication {replication_number} Finished*****************")
    log.info(f"Remaining Regions: {remaining_regions_l}")
    log.info(f"Classified Regions: {classified_region_l}")
    log.info(f"Unidentified Regions: {unidentified_regions_l}")
    log.info(f"Infeasible Regions: {infeasible_regions_l}")
    log.info(f"{tf_wrapper.count} Evaluations completed")
    log.info(f"{options.max_budget - tf_wrapper.count} left")
    log.info("**************************************************")
    
    total_time_elapsed = time.perf_counter() - total_time_start

    time_result = {"total_time": total_time_elapsed,
                   "simulation_time": np.sum(tf_wrapper.simultation_time),
                   "simulation_time_history": tf_wrapper.simultation_time, 
                   "total_non_simulation_time": total_time_elapsed - np.sum(tf_wrapper.simultation_time)}
    
    with open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME+ "_" + str(replication_number) + "_time.pkl"), "wb") as f:
        pickle.dump(time_result, f)

    with open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME+ "_" + str(replication_number) + ".pkl"), "wb") as f:
        pickle.dump(ftree,f)
    
    falsification_volume_arrays = fv_using_gp(ftree, options, oracle_info, options.fv_quantiles_for_gp, rng)

    with open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME + "_" + str(replication_number) + "_fal_val_gp.pkl"), "wb") as f:
        pickle.dump(falsification_volume_arrays,f)

    with open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME + "_" + str(replication_number) + "_point_history.pkl"), "wb") as f:
        pickle.dump(tf_wrapper.point_history, f)
    

    log.info("Ended {} Replication {} with seed {}".format(BENCHMARK_NAME, replication_number, seed))
    print("Ended replication {}".format(replication_number))
    log.removeHandler(fh)
    fh.close()

    return ftree, time_result, falsification_volume_arrays, tf_wrapper.point_history