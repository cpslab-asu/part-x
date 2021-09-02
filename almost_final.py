from classification import calculate_volume
from utils_partx import branch_new_region_support, pointsInSubRegion
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
number_of_samples_gen_GP = 100
R=10
M=1000
delta = 0.001
alpha = [0.95]
region_support = np.array([[[-1., 1.], [-1., 1.]]])
initialization_budget = 50
max_budget = 20000
number_of_BO_samples = [100]
continued_sampling_budget = 50
branching_factor = 2
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
queue_leaves_r = []
queue_leaves_c = []
queueLeaves_u = []
if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
    queue_leaves_r.append(id)
elif region_class == '+' or region_class == '-':
    queue_leaves_c.append(id)



# Tree initialization using root
ftree = Tree()
ftree.create_node(id,id,data=root)


while test_function.callCount <= options.max_budget:
    # Remaining Region Classification

    tempQueue = queue_leaves_r.copy()


    for temp_node_id in tempQueue:

        node = ftree.get_node(temp_node_id)
        parent = node.identifier
        node = node.data
        new_bounds = branch_new_region_support(node.region_support, direction[node.direction_of_branch % options.test_function_dimension], True, options.branching_factor)
        points_division_samples_in, points_division_samples_out = pointsInSubRegion(node.samples_in, node.samples_out, new_bounds)
        for iterate in range(new_bounds.shape[0]):
            id = id+1
            new_region_supp = np.reshape(new_bounds[iterate], (1,new_bounds[iterate].shape[0],new_bounds[iterate].shape[1]))

            new_node = partx_node(new_region_supp, points_division_samples_in[iterate], points_division_samples_out[iterate], (node.direction_of_branch+1), node.region_class)
            samples_in, samples_out = new_node.samples_management_unclassified(options)
            region_class = new_node.calculate_and_classifiy(options)
            ftree.create_node(id, id, parent = parent, data = new_node)
            
            if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                queue_leaves_r.append(id)
            elif region_class == '+' or region_class == '-':
                queue_leaves_c.append(id)
            elif region_class == 'u':
                queueLeaves_u.append(id)


# classified Region reclassification
    tempQueue = queue_leaves_c.copy()
    queue_leaves_c = []
    volumes = []
    for temp_node_id in tempQueue:
        node = ftree.get_node(temp_node_id)
        parent = node.identifier
        node = node.data
        volumes.append(calculate_volume(node.region_support)[0])

    volume_distribution = volumes/np.sum(volumes)
    arg_ind = np.argsort(-1*volume_distribution)
    
    tempQueue = np.take_along_axis(np.array(tempQueue), arg_ind, axis = 0)    
    volume_distribution = volume_distribution[arg_ind]
    rem_budget = float(initialization_budget)
    budget_remaining_ = []
    budget_assigned_ = []
    for iterate, temp_node_id in enumerate(tempQueue):
        node = ftree.get_node(temp_node_id)
        parent = node.identifier
        node = node.data
        if iterate == 0:
            assigned_budget = int(volume_distribution[iterate]*rem_budget)
            rem_budget = rem_budget - assigned_budget
            budget_assigned_.append(assigned_budget)
            budget_remaining_.append(rem_budget)

        else:
            assigned_budget = int(min(rem_budget,volume_distribution[iterate]*options.continued_sampling_budget))
            rem_budget = rem_budget - assigned_budget
            budget_assigned_.append(assigned_budget)
            budget_remaining_.append(rem_budget)
        
        samples_in, samples_out = node.samples_management_classified(options, assigned_budget)
        region_class = node.calculate_and_classifiy(options)
        
        if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
            queue_leaves_r.append(parent)
        elif region_class == '+' or region_class == '-':
            queue_leaves_c.append(parent)
        elif region_class == 'u':
            queueLeaves_u.append(parent)
    print("Budget Left = {}".format(options.max_budget-test_function.callCount))
    print("******************************")
    print("Region Unclassified = {}".format(len(queue_leaves_r)))
    print("Region Classified = {}".format(len(queue_leaves_c)))
    print("************************************")
    


print("Budget Used = {}".format(test_function.callCount))





from utils_partx import plotRegion

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

plt.show()