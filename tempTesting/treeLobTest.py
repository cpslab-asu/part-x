from tp import partx
# from tree import calculateArea
from treelib import Tree
import matplotlib.pyplot as plt

def calculateArea(regionBounds):
    dim_0_length = regionBounds[0][0][1] - regionBounds[0][0][0]
    dim_1_length = regionBounds[0][1][1] - regionBounds[0][1][0]

    return dim_0_length * dim_1_length

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

ftree = Tree()
ftree.create_node(id,id,data = partx(regionBounds))
queueLeaves.append(id)

# root = Tree('Root')
# child01 = Tree('C01')
# root.addChild(child01)
# child01 = Tree('C02')
# root.addChild(child01)

#add children nodes
leftFlag = 0
rightFlag = 0
print("*******************************************************************")
while (len(queueLeaves) != 0):
    leftFlag = 0
    rightFlag = 0
    temp = queueLeaves.pop(0)
    temp_node = ftree.get_node(temp)
    # print(temp_node.identifier)
    new_bound_0, new_bound_1 = branchingCondition(temp_node.data)
    
    parent = temp_node.identifier
    # print(calculateArea(new_bound_0))
    # print(calculateArea(new_bound_1))
    if calculateArea(new_bound_0) >= eps:

        id = id+1
        # print("********")
        ftree.create_node(id, id, parent = parent, data = partx(new_bound_0))

        queueLeaves.append(id)
        fig = plotRegion(new_bound_0)
        
        # print(id)
        # print("adding left child")


    if calculateArea(new_bound_1) >= eps:
        id = id+1
        # print("********")
        ftree.create_node(id, id, parent = parent, data = partx(new_bound_1))
        queueLeaves.append(id)
        fig = plotRegion(new_bound_1)
        
        # print(id)
        # print("adding right child")
    
    

    # print(len(queue_leaves))
    # print("**********************************************************************")
    # print("**********************************************************************")

ftree.show()
print(ftree.depth())

# plt.show()

leaves = ftree.leaves()
leaf_identifier = []
leaf_areas = []
for i in leaves:
    leaf_identifier.append(i.identifier)
    leaf_areas.append(i.data.area)
print(leaf_identifier)
print(leaf_areas)