import numpy as np
from numpy.lib.shape_base import expand_dims

class partx(object):
    def __init__(self, regionBounds, expandLastTime = 0, terminate = 0):
        # self.key = key
        self.regionBounds = regionBounds
        # self.randNum = np.ndarray.item(np.random.uniform(0,1,1))
        self.randNum = 0.5
        self.direction = np.ndarray.item(np.random.randint(0,2,1))
        self.area = self.calculateBoundArea()

    
    def calculateBoundArea(self):
        dim_0_length = self.regionBounds[0][0][1] - self.regionBounds[0][0][0]
        dim_1_length = self.regionBounds[0][1][1] - self.regionBounds[0][1][0]
        # print(dim_0_length * dim_1_length)
        return dim_0_length * dim_1_length