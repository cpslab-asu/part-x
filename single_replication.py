from classification import calculate_volume
from utils_partx import assign_budgets, branch_new_region_support, pointsInSubRegion
from testFunction import callCounter
from partx_node import partx_node
from partx_options import partx_options
import numpy as np
from classification import calculate_volume
import matplotlib.pyplot as plt
from budget_check import budget_check
from treelib import Tree
from calIntegral import calculate_mc_integral
from utils_partx import plotRegion
import logging
import pickle
""""
# add nugget effect to MC_integral_function
# nugget is absolute of normal distributio with mean 0 and var 0.001

# Ensemble part
"""

def run_single_replication(inputs):
    q, options, exp_name, seed, test_function = inputs

    callCounts = callCounter(test_function)
    
    log = logging.getLogger()
    log.setLevel(logging.INFO) 
    fh = logging.FileHandler(filename="logging_info/"+exp_name+".log")
    formatter = logging.Formatter(
                    fmt='%(message)s'
                    )
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.info("Information about Replication {}".format(q))
    log.info("**************************************************")
    log.info("Initial Values for Replication {}")

    log.info("Budget Used = {}".format(callCounts.callCount))
    log.info("Budget Available (Max Budget) = {}".format(options.max_budget))
    log.info("**************************************************")
    print("Started replication {}".format(q))

    rng = np.random.default_rng(seed)
# Sampling Initialization
    samples_in = np.array([[[]]])
    samples_out = np.array([[]])
    direction = [1,0]

    direction_of_branch = 0



    # define root node
    root = partx_node(options.initial_region_support, samples_in, samples_out, direction_of_branch)
    samples_in, samples_out = root.samples_management_unclassified(options, callCounts, rng)
    region_class = root.calculate_and_classifiy(options,rng)

    id = 0
    remaining_regions_list = []
    classified_regions_list = []
    queueLeaves_u = []
    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
        remaining_regions_list.append(id)
    elif region_class == '+' or region_class == '-':
        classified_regions_list.append(id)
    elif region_class == 'u':
        queueLeaves_u.append(id)



    # Tree initialization using root
    ftree = Tree()
    ftree.create_node(id,id,data=root)


    while budget_check(options, callCounts.callCount, remaining_regions_list):
        tempQueue = remaining_regions_list.copy()
        remaining_regions_list = []

        for temp_node_id in tempQueue:
            node = ftree.get_node(temp_node_id)
            parent = node.identifier
            node = node.data
            new_bounds = branch_new_region_support(node.region_support, direction[node.direction_of_branch % options.test_function_dimension], options.uniform_partitioning, options.branching_factor, rng)
            points_division_samples_in, points_division_samples_out = pointsInSubRegion(node.samples_in, node.samples_out, new_bounds)
            for iterate in range(new_bounds.shape[0]):
                # print("parent = {} ------> id{}".format(parent,id))
                id = id+1
                new_region_supp = np.reshape(new_bounds[iterate], (1,new_bounds[iterate].shape[0],new_bounds[iterate].shape[1]))

                new_node = partx_node(new_region_supp, points_division_samples_in[iterate], points_division_samples_out[iterate], (node.direction_of_branch+1), node.region_class)
                samples_in, samples_out = new_node.samples_management_unclassified(options, callCounts, rng)
                region_class = new_node.calculate_and_classifiy(options,rng)
                ftree.create_node(id, id, parent = parent, data = new_node)
                
                if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                    remaining_regions_list.append(id)
                elif region_class == '+' or region_class == '-':
                    classified_regions_list.append(id)
                elif region_class == 'u':
                    queueLeaves_u.append(id)
        log.info("**************************************************")
        log.info("Mid Classified regions = {}".format(len(classified_regions_list)))
        log.info("Mid Unclassified regions = {}".format(len(remaining_regions_list)))
        log.info("Mid Unidentified regions = {}".format(len(queueLeaves_u)))
        if len(classified_regions_list) != 0:

            tempQueue = classified_regions_list.copy()
            classified_regions_list = []
            volumes = []
            #original
            # for temp_node_id in tempQueue:
            #     node = ftree.get_node(temp_node_id)
            #     parent = node.identifier
            #     node = node.data
            #     volumes.append(calculate_volume(node.region_support)[0])


            # volume_distribution = volumes/np.sum(volumes)
            # End original

            for temp_node_id in tempQueue:
                node = ftree.get_node(temp_node_id)
                parent = node.identifier
                node = node.data
                cdf_sum = calculate_mc_integral(node.samples_in, node.samples_out, node.region_support, options.test_function_dimension, options.R, options.M, rng)
                volumes.append((cdf_sum * calculate_volume(node.region_support)[0]) + abs(rng.normal(options.nugget_mean, options.nugget_std_dev))) # + nugget
            
            if np.sum(volumes) != 0.0:
                volume_distribution = volumes/np.sum(volumes)
            else:
                volume_distribution = volumes
            
            n_cont_sampling_budget_assignment = assign_budgets(volume_distribution, options.continued_sampling_budget, rng)

            for iterate, temp_node_id in enumerate(tempQueue):
                if n_cont_sampling_budget_assignment[iterate] != 0:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    node = node.data
                    
                    samples_in, samples_out = node.samples_management_classified(options, callCounts, n_cont_sampling_budget_assignment[iterate], rng)

                    region_class = node.calculate_and_classifiy(options, rng)
                    
                    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                        remaining_regions_list.append(parent)
                    elif region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        queueLeaves_u.append(parent)
                else:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    region_class = node.data.region_class
                    
                    if region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)

        log.info("Classified regions = {}".format(len(classified_regions_list)))
        log.info("Unclassified regions = {}".format(len(remaining_regions_list)))
        log.info("Unidentified regions = {}".format(len(queueLeaves_u)))
        log.info("Budget available for replication = {}".format(options.max_budget - callCounts.callCount))
        log.info("**************************************************")
    budget_available = options.max_budget - callCounts.callCount
    if budget_available >= 0:
        log.info("**************************************************")
        log.info("In the last act for replication {}:".format(q))
        log.info("Budget Used for replication {} = {}".format(q, callCounts.callCount))
        log.info("Budget available for replication {} = {}".format(q, budget_available))
        log.info("**************************************************")

        tempQueue = classified_regions_list + remaining_regions_list
        if len(tempQueue) != 0:
            remaining_regions_list = []
            classified_regions_list = []
            volumes = []
            for temp_node_id in tempQueue:
                node = ftree.get_node(temp_node_id)
                parent = node.identifier
                node = node.data
                cdf_sum = calculate_mc_integral(node.samples_in, node.samples_out, node.region_support, options.test_function_dimension, options.R, options.M,rng)
                volumes.append((cdf_sum * calculate_volume(node.region_support)[0]) + abs(rng.normal(options.nugget_mean, options.nugget_std_dev))) # + nugget

            volume_distribution = volumes/np.sum(volumes)
            n_cont_sampling_budget_assignment = assign_budgets(volume_distribution,budget_available,rng)

            for iterate, temp_node_id in enumerate(tempQueue):
                if n_cont_sampling_budget_assignment[iterate] != 0:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    node = node.data

                    samples_in, samples_out = node.samples_management_classified(options, callCounts, n_cont_sampling_budget_assignment[iterate],rng)

                    region_class = node.calculate_and_classifiy(options,rng)
                    
                    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                        remaining_regions_list.append(parent)
                    elif region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        queueLeaves_u.append(parent)
                else:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    region_class = node.data.region_class
                    
                    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                        remaining_regions_list.append(parent)
                    elif region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        queueLeaves_u.append(parent)

    budget_available = options.max_budget - callCounts.callCount
    log.info("**************************************************")
    log.info("**************************************************")
    log.info("ENDING replication {}:".format(q))
    log.info("Budget Used = {}".format(callCounts.callCount))
    log.info("Budget available = {}".format(budget_available))
    log.info("**************************************************")
    log.info("**************************************************")

    f = open(exp_name + ".pkl", "wb")
    pickle.dump(ftree,f)
    f.close()

    print("Ended replication {}".format(q))
    log.removeHandler(fh)
    del log, fh

    return [ftree, classified_regions_list, remaining_regions_list, queueLeaves_u]
    