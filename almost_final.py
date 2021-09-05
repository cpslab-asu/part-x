from classification import calculate_volume
from utils_partx import assign_budgets, branch_new_region_support, pointsInSubRegion
from testFunction import test_function
from partx_node import partx_node
from partx_options import partx_options
import numpy as np
from classification import calculate_volume
import matplotlib.pyplot as plt
from budget_check import budget_check
from treelib import Tree

# Options initialization
test_function_dimension = 2
delta = 0.001
alpha = [0.95]
region_support = np.array([[[-1., 1.], [-1., 1.]]])

initialization_budget = 10
max_budget = 5000
number_of_BO_samples = [20]
number_of_samples_gen_GP = 500
continued_sampling_budget = 100
branching_factor = 2

R = number_of_BO_samples[0]
M = number_of_samples_gen_GP

options = partx_options(region_support, branching_factor, test_function_dimension, 
                        number_of_BO_samples, number_of_samples_gen_GP, alpha, M, R, 
                        delta, True, initialization_budget, max_budget, 
                        continued_sampling_budget)


# Sampling Initialization
samples_in = np.array([[[]]])
samples_out = np.array([[]])
direction = [1,0]

direction_of_branch = 0



# define root node
root = partx_node(region_support, samples_in, samples_out, direction_of_branch)
samples_in, samples_out = root.samples_management_unclassified(options)
region_class = root.calculate_and_classifiy(options)

id = 0
remaining_regions_list = []
classified_regions_list = []
queueLeaves_u = []
if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
    remaining_regions_list.append(id)
elif region_class == '+' or region_class == '-':
    classified_regions_list.append(id)



# Tree initialization using root
ftree = Tree()
ftree.create_node(id,id,data=root)


while budget_check(options, test_function.callCount, remaining_regions_list):
    # Remaining Region Classification
    
    tempQueue = remaining_regions_list.copy()
    remaining_regions_list = []

    for temp_node_id in tempQueue:
    # while len(tempQueue) != 0:
    #     temp_node_id = tempQueue.pop(0)
        node = ftree.get_node(temp_node_id)
        parent = node.identifier
        node = node.data
        new_bounds = branch_new_region_support(node.region_support, direction[node.direction_of_branch % options.test_function_dimension], True, options.branching_factor)
        points_division_samples_in, points_division_samples_out = pointsInSubRegion(node.samples_in, node.samples_out, new_bounds)
        for iterate in range(new_bounds.shape[0]):
            # print("parent = {} ------> id{}".format(parent,id))
            id = id+1
            new_region_supp = np.reshape(new_bounds[iterate], (1,new_bounds[iterate].shape[0],new_bounds[iterate].shape[1]))

            new_node = partx_node(new_region_supp, points_division_samples_in[iterate], points_division_samples_out[iterate], (node.direction_of_branch+1), node.region_class)
            samples_in, samples_out = new_node.samples_management_unclassified(options)
            region_class = new_node.calculate_and_classifiy(options)
            ftree.create_node(id, id, parent = parent, data = new_node)
            
            if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                remaining_regions_list.append(id)
            elif region_class == '+' or region_class == '-':
                classified_regions_list.append(id)
            elif region_class == 'u':
                queueLeaves_u.append(id)
    # print(remaining_regions_list)
############################################################################################################
# classified Region reclassification
    if len(classified_regions_list) != 0:
        # print(test_function.callCount)
        # print("predicted next = {}".format(test_function.callCount + options.continued_sampling_budget))
        tempQueue = classified_regions_list.copy()
        classified_regions_list = []
        volumes = []
        for temp_node_id in tempQueue:
            node = ftree.get_node(temp_node_id)
            parent = node.identifier
            node = node.data
            volumes.append(calculate_volume(node.region_support)[0])

        volume_distribution = volumes/np.sum(volumes)
        # print("Volume distribution is {}".format(volume_distribution))
        n_cont_sampling_budget_assignment = assign_budgets(volume_distribution, options.continued_sampling_budget)
        # print("total Ccont budget confirmation = {}".format(np.sum(n_cont_sampling_budget_assignment)))
        if len(n_cont_sampling_budget_assignment) != 0:
            # print("is it true? {}".format(test_function.callCount))
            for iterate, temp_node_id in enumerate(tempQueue):
                if n_cont_sampling_budget_assignment[iterate] != 0:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    node = node.data
                    # print("Haha")
                    # print(test_function.callCount)
                    samples_in, samples_out = node.samples_management_classified(options, n_cont_sampling_budget_assignment[iterate])
                    # print(test_function.callCount)
                    # print("Haha")
                    region_class = node.calculate_and_classifiy(options)
                    
                    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                        remaining_regions_list.append(parent)
                    elif region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        queueLeaves_u.append(parent)

            # print("check here {}".format(test_function.callCount))
        # print(classified_regions_list)
        

############################################################################################################
# Last Act - Region reclassification
budget_available = options.max_budget - test_function.callCount
print("**********************************************")
print("**********************************************")
print("In the last act:")
print("Budget Used = {}".format(test_function.callCount))
print("Budget available = {}".format(budget_available))
print("**********************************************")
print("**********************************************")
tempQueue = classified_regions_list + remaining_regions_list
if len(tempQueue) != 0:
        # print(test_function.callCount)
        # print("predicted next = {}".format(test_function.callCount + options.continued_sampling_budget))
        remaining_regions_list = []
        classified_regions_list = []
        volumes = []
        for temp_node_id in tempQueue:
            node = ftree.get_node(temp_node_id)
            parent = node.identifier
            node = node.data
            volumes.append(calculate_volume(node.region_support)[0])

        volume_distribution = volumes/np.sum(volumes)
        # print("Volume distribution is {}".format(volume_distribution))
        n_cont_sampling_budget_assignment = assign_budgets(volume_distribution,budget_available)
        # print("total Ccont budget confirmation = {}".format(np.sum(n_cont_sampling_budget_assignment)))
        if len(n_cont_sampling_budget_assignment) != 0:
            # print("is it true? {}".format(test_function.callCount))
            for iterate, temp_node_id in enumerate(tempQueue):
                if n_cont_sampling_budget_assignment[iterate] != 0:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    node = node.data
                    # print("Haha")
                    # print(test_function.callCount)
                    samples_in, samples_out = node.samples_management_classified(options, n_cont_sampling_budget_assignment[iterate])
                    # print(test_function.callCount)
                    # print("Haha")
                    region_class = node.calculate_and_classifiy(options)
                    
                    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                        remaining_regions_list.append(parent)
                    elif region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        queueLeaves_u.append(parent)



budget_available = options.max_budget - test_function.callCount
print("**********************************************")
print("**********************************************")
print("ENDING:")
print("Budget Used = {}".format(test_function.callCount))
print("Budget available = {}".format(budget_available))
print("**********************************************")
print("**********************************************")


from utils_partx import plotRegion
ftree.show()

import pickle
f = open("tree.pkl", "wb")
pickle.dump(ftree,f)

leaves = ftree.leaves()
print("number of leaves= {}".format(len(leaves)))
print("******************")
# plt.ion()

print("*******************************************************")
for x,i in enumerate(leaves):
    # fig = plt.figure()
    x_1, y_1, x_2,y_2,x_3,y_3,x_4,y_4 = plotRegion(i.data.region_support)
    plt.plot(x_1,y_1)
    plt.plot(x_2,y_2)
    plt.plot(x_3,y_3)
    plt.plot(x_4,y_4)

    
    if i.data.region_class == "+":
        plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'g.')
    elif i.data.region_class == "-":
        plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'r.')
plt.title("Rosenbrock Function Budget = {} -- BO Grid {} x {}".format(options.max_budget, number_of_BO_samples[0], number_of_samples_gen_GP))
plt.show()

print("final budget check = {}".format(test_function.callCount))