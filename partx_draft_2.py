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
max_budget = 25000
number_of_BO_samples = [50]

options = partx_options(region_support, test_function_dimension, number_of_BO_samples, number_of_samples_gen_GP, alpha, M, R, delta, True, initialization_budget, max_budget)

samples_in = np.array([[[]]])
samples_out = np.array([[]])
direction = [0,1]

direction_of_branch = 0
# number_of_samples = 5


root = partx_node(region_support, samples_in, samples_out, direction_of_branch)
samples_in, samples_out = root.samples_management_unclassified(options)
region_class = root.calculate_and_classifiy(options)

id = 0
queueLeaves = []

queueLeaves.append(id)

ftree = Tree()
ftree.create_node(id,id,data=root)
unidentified = []
while test_function.callCount <= options.max_budget:
    unclassified = []
    classified = []

    for node_number in queueLeaves:
        node = ftree.get_node(node_number)
        
        if node.data.region_class == 'r' or node.data.region_class == 'r+' or node.data.region_class == 'r-':
            unclassified.append(node)
        elif node.data.region_class == '+' or node.data.region_class == '-':
            classified.append(node)
        else:
            unidentified.append(node)

    print("Unclassified = {}".format(len(unclassified)))
    print("Classified = {}".format(len(classified)))
    queueLeaves = []

    while len(unclassified)!=0:
        temp_node = unclassified.pop(0)
        
        # print("Popped up: {}".format(tempQueue))
        node_data = temp_node.data
        parent = temp_node.identifier
        

        new_bounds = branch_new_region_support(node_data.region_support, direction[node_data.direction_of_branch % 2], True, 2)
        points_division_samples_in, points_division_samples_out = pointsInSubRegion(node_data.samples_in, node_data.samples_out, new_bounds)
        for iterate in range(new_bounds.shape[0]):
            id = id+1
            # print("parent = {} ------> id{}".format(parent,id))
            new_region_supp = np.reshape(new_bounds[iterate], (1,new_bounds[iterate].shape[0],new_bounds[iterate].shape[1]))

            new_node = partx_node(new_region_supp, points_division_samples_in[iterate], points_division_samples_out[iterate], (node_data.direction_of_branch+1))

            samples_in, samples_out = new_node.samples_management_unclassified(options)
            region_class = new_node.calculate_and_classifiy(options)
            ftree.create_node(id, id, parent = parent, data = new_node)
            
            queueLeaves.append(id)

            
    
    volumes = []
    for temp_node in classified:
        volumes.append(calculate_volume(temp_node.data.region_support)[0])
    
    volume_distribution = np.array(np.round((np.array(volumes) / np.sum(np.array(volumes))) * number_of_BO_samples), dtype = np.int)
    volume_distribution_iterate = 0

    while len(classified)!=0:

        temp_node = classified.pop(0)
        
        # print("Popped up: {}".format(tempQueue))
        node_data = temp_node.data
        node_id = temp_node.identifier
        #  create probablity based sampling points
        
        samples_in, samples_out = node_data.samples_management_classified(options, volume_distribution[volume_distribution_iterate])
        region_class = node_data.calculate_and_classifiy(options)
        
        queueLeaves.append(node_id)


    print("Queue Leaves = {}".format(queueLeaves))

    print("Budget Left = {}".format(options.max_budget-test_function.callCount))
# print("Budget Used = {}".format(test_function.callCount))
# print("***********************")
# print(classified_nodes)
# print("************************")
ftree.show()
print(ftree.depth())

from utils_partx import plotRegion

leaves = ftree.leaves()
print("number of leaves= {}".format(len(leaves)))
print("******************")
plt.show()
for i in leaves:
    x_1, y_1, x_2,y_2,x_3,y_3,x_4,y_4 = plotRegion(i.data.region_support)
    plt.plot(x_1,y_1)
    plt.plot(x_2,y_2)
    plt.plot(x_3,y_3)
    plt.plot(x_4,y_4)
    # print(i)
    # print(i.data.region_class)
    if i.data.region_class == 'r' or i.data.region_class == 'r+' or i.data.region_class == 'r-':
        plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'b.')
    
    if i.data.region_class == '+' or i.data.region_class == '-':
        print(i.data.region_support)
        plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'r.')

plt.show()