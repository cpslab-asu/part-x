from tree import calculateArea
from tree_temp import Tree
import numpy as np
import matplotlib.pyplot as plt

def plotRegion(regionBounds):
    x_coordinates_1 = [regionBounds[0][0][0], regionBounds[0][0][0]]
    y_coordinates_1 = [regionBounds[0][1][0], regionBounds[0][1][1]]

    x_coordinates_2 = [regionBounds[0][0][0], regionBounds[0][0][1]]
    y_coordinates_2 = [regionBounds[0][1][0], regionBounds[0][1][0]]

    x_coordinates_3 = [regionBounds[0][0][1], regionBounds[0][0][1]]
    y_coordinates_3 = [regionBounds[0][1][0], regionBounds[0][1][1]]

    x_coordinates_4 = [regionBounds[0][0][0], regionBounds[0][0][1]]
    y_coordinates_4 = [regionBounds[0][1][1], regionBounds[0][1][1]]

    
    # print(x_coordinates)
    # print(y_coordinates)
    plt.plot(x_coordinates_1, y_coordinates_1)
    plt.plot(x_coordinates_2, y_coordinates_2)
    plt.plot(x_coordinates_3, y_coordinates_3)
    plt.plot(x_coordinates_4, y_coordinates_4)

    return plt
   
def branchingCondition(node):
        dim_0_length = node.regionBounds[0][0][1] - node.regionBounds[0][0][0]
        dim_1_length = node.regionBounds[0][1][1] - node.regionBounds[0][1][0]

        if node.direction == 0:
            new_bound_0 = [[[node.regionBounds[0][0][0], node.regionBounds[0][0][0] + dim_0_length * node.randNum],
                            node.regionBounds[0][1]]]
            new_bound_1 = [[[node.regionBounds[0][0][0] + dim_0_length * node.randNum, node.regionBounds[0][0][1]],
                            node.regionBounds[0][1]]]
        elif node.direction == 1:
            new_bound_0 = [[node.regionBounds[0][0],
                        [node.regionBounds[0][1][0], node.regionBounds[0][1][0] + dim_1_length * node.randNum]]]
            new_bound_1 = [[node.regionBounds[0][0],
                        [node.regionBounds[0][1][0] + dim_1_length * node.randNum, node.regionBounds[0][1][1]]]]
        # print("Original Bounds are: {}".format(node.regionBounds[0]))
        # print(node.direction)
        # print("Bounds 1 are: {} \t\t Area = {}".format(new_bound_0,calculateArea(new_bound_0)))
        # print("Bounds 2 are: {} \t\t Area = {}".format(new_bound_1,calculateArea(new_bound_1)))
        # print("**************************************************************************")
        return new_bound_0, new_bound_1


eps = 0.01
regionBounds = [[[0,4],[0,4]]]
id = 1
queueLeaves = []

root = Tree(regionBounds, id)
queueLeaves.append(root)

# root = Tree('Root')
# child01 = Tree('C01')
# root.addChild(child01)
# child01 = Tree('C02')
# root.addChild(child01)

#add children nodes
leftFlag = 0
rightFlag = 0
while (len(queueLeaves) != 0):
    leftFlag = 0
    rightFlag = 0
    temp = queueLeaves.pop(0)
    new_bound_0, new_bound_1 = branchingCondition(temp)
    
    
    if calculateArea(new_bound_0) >= eps:
        id = id+1
        leftChild = Tree(new_bound_0, id)
        queueLeaves.append(leftChild)
        temp.addChild(leftChild)
        fig = plotRegion(new_bound_0)
        
        # print(id)
        # print("adding left child")
    


    if calculateArea(new_bound_1) >= eps:
        id = id+1
        rightChild = Tree(new_bound_1, id)
        queueLeaves.append(rightChild)
        temp.addChild(rightChild)
        fig = plotRegion(new_bound_1)
        
        # print(id)
        # print("adding right child")
    

    # print(len(queue_leaves))
    # print("**********************************************************************")
    # print("**********************************************************************")

root.prettyTree()

plt.show()


print(Tree.getChild(Tree.getChild(root,1),0))