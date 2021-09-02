from classification import calculate_volume
from utils_partx import branch_new_region_support, pointsInSubRegion
from testFunction import test_function
from partx_node import partx_node
from partx_options import partx_options
import numpy as np
from classification import calculate_volume
import matplotlib.pyplot as plt
from treelib import Tree


test_function_dimension = 2
number_of_samples_gen_GP = 100
R=10
M=1000
delta = 0.001
alpha = [0.95]
region_support = np.array([[[-1., 1.], [-1., 1.]]])
initialization_budget = 50
max_budget = 3000
number_of_BO_samples = [100]
continued_sampling_budget = 50
options = partx_options(region_support, test_function_dimension, number_of_BO_samples, number_of_samples_gen_GP, alpha, M, R, delta, True, initialization_budget, max_budget, continued_sampling_budget)

samples_in = np.array([[[]]])
samples_out = np.array([[]])
direction = [1,0]

direction_of_branch = 0
# number_of_samples = 5


root = partx_node(region_support, samples_in, samples_out, direction_of_branch)
samples_in, samples_out = root.samples_management_unclassified(options)
region_class = root.calculate_and_classifiy(options)

id = 0
queueLeaves_r = []
queueLeaves_c = []

if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
    queueLeaves_r.append(id)
elif region_class == '+' or region_class == '-':
    queueLeaves_c.append(id)

classified_nodes = []
ftree = Tree()
ftree.create_node(id,id,data=root)
# print(queueLeaves_r)
# print("*********************")
# print(len(queueLeaves_c)+len(queueLeaves_r))
while test_function.callCount <= options.max_budget:
    # print(len(queueLeaves_c))
    # print(len(queueLeaves_r))
    # print("***********************************")
    tempQueue = queueLeaves_r.copy()
    queueLeaves = []
    # print(tempQueue)
    uc = 0
    c = 0
    while len(tempQueue)!=0:
        uc = uc+1
        # print("unclassified: {}".format(uc))
        # print("******")
        temp_node = tempQueue.pop(0)

        # print("Popped up: {}".format(tempQueue))
        node = ftree.get_node(temp_node)
        parent = node.identifier
        node = node.data
        # if node.region_class == "r" or node.region_class == "r+" or node.region_class == "r-":
        # print(direction[node.direction_of_branch % 2])
        new_bounds = branch_new_region_support(node.region_support, direction[node.direction_of_branch % 2], True, 2)
        points_division_samples_in, points_division_samples_out = pointsInSubRegion(node.samples_in, node.samples_out, new_bounds)
        for iterate in range(new_bounds.shape[0]):
            id = id+1
            # print("parent = {} ------> id{}".format(parent,id))
            new_region_supp = np.reshape(new_bounds[iterate], (1,new_bounds[iterate].shape[0],new_bounds[iterate].shape[1]))

            new_node = partx_node(new_region_supp, points_division_samples_in[iterate], points_division_samples_out[iterate], (node.direction_of_branch+1), node.region_class)
            # print(node.samples_in)
            samples_in, samples_out = new_node.samples_management_unclassified(options)
            region_class = new_node.calculate_and_classifiy(options)
            ftree.create_node(id, id, parent = parent, data = new_node)
            
            if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                # print("Appending id: {}".format(id))
                queueLeaves_r.append(id)
            if region_class == '+' or region_class == '-':
                print("Added from normal classification: {}".format(id))
                # print("Something classified for first time")
                # print(vars(new_node))
                # print("********************************************")
                queueLeaves_c.append(id)

    tempQueue = queueLeaves_c.copy()
    print(tempQueue)
    # print(tempQueue)
    queueLeaves_c = []
    volumes = []
    for temp_node_id in tempQueue:
        node = ftree.get_node(temp_node_id)
        parent = node.identifier
        node = node.data
        volumes.append(calculate_volume(node.region_support)[0])
    
    
    
    # volume_distribution = np.array(np.round((np.array(volumes) / np.sum(np.array(volumes))) * number_of_BO_samples), dtype = np.int)
    # volume_distribution_iterate = 0
    # while len(tempQueue)!=0:
    #     c = c+1
    #     # print("classified: {}".format(c))
    #     # print("******")
    #     temp_node = tempQueue.pop(0)
    #     node = ftree.get_node(temp_node)
    #     parent = node.identifier
    #     node = node.data
    #     # print(node.region_support)
    #     #  create probablity based sampling points
        
    #     samples_in, samples_out = node.samples_management_classified(options, max(volume_distribution[volume_distribution_iterate],1))
    #     volume_distribution_iterate = volume_distribution_iterate+1
    #     region_class = node.calculate_and_classifiy(options)
        
    #     if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
    #         # print("Something reclassified into remaininng")
    #         # print(vars(node))
    #         # print("********************************************")
    #         queueLeaves_r.append(parent)
    #     if region_class == '+' or region_class == '-':
    #         print("Added from reclassification: {}".format(parent))
    #         # classified_nodes.append(id)
    #         # print("Something classified for again")
    #         # print(vars(node))
    #         # print("********************************************")
    #         queueLeaves_c.append(parent)

    volume_distribution = volumes/np.sum(volumes)
    arg_ind = np.argsort(-1*volume_distribution)
    

    tempQueue = np.take_along_axis(np.array(tempQueue), arg_ind, axis = 0)    
    volume_distribution = volume_distribution[arg_ind]
    print("***********************************")
    print("Volume distribution: {}".format(volume_distribution))
    rem_budget = float(initialization_budget)
    budget_remaining_ = []
    budget_assigned_ = []
    for iterate, temp_node_id in enumerate(tempQueue):
        # print("classified: {}".format(c))
        # print("******")
        node = ftree.get_node(temp_node)
        parent = node.identifier
        node = node.data
        # print(node.region_support)
        #  create probablity based sampling points
        if iterate == 0:
            assigned_budget = int(volume_distribution[iterate]*rem_budget)
            rem_budget = rem_budget - assigned_budget
            budget_assigned_.append(assigned_budget)
            budget_remaining_.append(rem_budget)

        else:
            assigned_budget = int(min(rem_budget,volume_distribution[iterate]*initialization_budget))
            rem_budget = rem_budget - assigned_budget
            budget_assigned_.append(assigned_budget)
            budget_remaining_.append(rem_budget)
        
        samples_in, samples_out = node.samples_management_classified(options, assigned_budget)
        # volume_distribution_iterate = volume_distribution_iterate+1
        region_class = node.calculate_and_classifiy(options)
        
        if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
            # print("Something reclassified into remaininng")
            # print(vars(node))
            # print("********************************************")
            queueLeaves_r.append(parent)
        if region_class == '+' or region_class == '-':
            print("Added from reclassification: {}".format(parent))
            # classified_nodes.append(id)
            # print("Something classified for again")
            # print(vars(node))
            # print("********************************************")
            queueLeaves_c.append(parent)
    print("********************")
    print(budget_remaining_)
    print(budget_assigned_)
    print("******************************************")
    # queueLeaves_c = tempQueue
    # tempQueue = []
    print("Budget Left = {}".format(options.max_budget-test_function.callCount))
print("Budget Used = {}".format(test_function.callCount))
print("***********************")
print(classified_nodes)
print("************************")
# ftree.show()
# print(ftree.depth())

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

    # print(i)
    # print(i.data.region_class)
    
    if i.data.region_class == "+":
        # print(x)
        # print(i.data.region_support)
        # print(vars(i.data))
        plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'g.')
    elif i.data.region_class == "-":
        # print(x)
        # print(i.data.region_support)
        # print(vars(i.data))
        plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'r.')
    # elif i.data.region_class == 'r' or i.data.region_class == 'r+' or i.data.region_class == 'r-':
    #     plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'b.')
        # fig_2.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'b.')
    
    
        # fig_2.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'y.')
    # fig.savefig(str(x)+".png")
    # fig.canvas.draw()
    # fig.canvas.flush_events()

plt.show()