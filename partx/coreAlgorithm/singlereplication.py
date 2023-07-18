
import numpy as np
import logging
from treelib import Tree
import time
import pickle
from copy import deepcopy

from ..results import fv_using_gp
from ..utils import Fn, branch_region, divide_points, calculate_volume
from .partx_node import PartXNode
from ..numerical import calculate_mc_integral, assign_budgets
from ..results import fv_using_gp


def run_single_replication(inputs):
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
    log.info("Information about Replication {}".format(replication_number))
    log.info("Running {} Replication {} with seed {}".format(BENCHMARK_NAME, replication_number, seed))
    log.info("**************************************************")
    log.info("Options File:")
    options_results = vars(options)
    for key, value in options_results.items():
        log.info("{} : {}".format(key, value))
    log.info("**************************************************")
    log.info("Budget Used = {}".format(tf_wrapper.count))
    log.info("Budget Available (Max Budget) = {}".format(options.max_budget))
    log.info("**************************************************")
    log.info("**************************************************")
    log.info("***************Replication Start******************")
    print("Started replication {}".format(replication_number))


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
                node = ftree.get_node(node_id)
                node_identifier = node.identifier
                node_data = node.data
                sub_bounds = branch_region(node_data.region_support, direction[node_data.branch_dir%options.tf_dim], options.uniform_partitioning, options.branching_factor, rng)
                x_samples_divided, y_samples_divided = divide_points(node_data.samples_in, node_data.samples_out, sub_bounds)
                
                for branches in range(options.branching_factor): 
                    node_id_keeper += 1

                    child_reg_sup = sub_bounds[branches]
                    child_reg_samples_in = x_samples_divided[branches]
                    child_reg_samples_out = y_samples_divided[branches]
                    budget_for_branching += max(options.init_budget - child_reg_samples_in.shape[0], 0) + options.bo_budget
                    child_node = deepcopy(PartXNode(node_id_keeper, node_identifier, child_reg_sup, child_reg_samples_in, child_reg_samples_out, node_data.branch_dir+1, region_class="r"))
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
                    node = ftree.get_node(classi_node)
                    node_data = node.data
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
                        node = ftree.get_node(classi_node)
                        node_identifier = node.identifier
                        node_data = node.data

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
                        node = ftree.get_node(classi_node)
                        node_identifier = node.identifier
                        node_data = node.data
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
                    node = ftree.get_node(all_nodes)
                    node_data = node.data
                    cs_indicator = calculate_volume(node_data.region_support)
                    volumes.append(cs_indicator)
                
                if np.sum(volumes) != 0.0:
                    volume_distribution = volumes/np.sum(volumes)
                else:
                    volume_distribution = volumes
                
                assigned_budgets = assign_budgets(volume_distribution, budget_left)

                for iterate, all_nodes in enumerate(all_regions):
                    
                    if assigned_budgets[iterate] != 0:
                        node = ftree.get_node(all_nodes)
                        node_identifier = node.identifier
                        node_data = node.data

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
                        node = ftree.get_node(all_nodes)
                        node_identifier = node.identifier
                        node_data = node.data
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

    return {
        'ftree': ftree,
        'time_results': time_result
    }
