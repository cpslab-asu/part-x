import numpy as np

class Node:
    def __init__(self, key, regionBounds):
        self.left = None
        self.right = None
        self.key = key
        self.regionBounds = regionBounds
        # self.randNum = np.ndarray.item(np.random.uniform(0,1,1))
        self.randNum = 0.5
        self.direction = np.ndarray.item(np.random.randint(0,2,1))
        self.area = self.calculateBoundArea()


    def calculateBoundArea(self):
        dim_0_length = self.regionBounds[0][0][1] - self.regionBounds[0][0][0]
        dim_1_length = self.regionBounds[0][1][1] - self.regionBounds[0][1][0]

        return dim_0_length * dim_1_length
        

def printTree(node, level=0):
    if node != None:
        printTree(node.left, level + 1)
        print(' ' * 4 * level + '->', node.key)
        printTree(node.right, level + 1)

    
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

def calculateArea(regionBounds):
    dim_0_length = regionBounds[0][0][1] - regionBounds[0][0][0]
    dim_1_length = regionBounds[0][1][1] - regionBounds[0][1][0]

    return dim_0_length * dim_1_length