import matplotlib.pyplot as plt
from ..numerical.sampling import lhs_sampling
import numpy as np
# from calculate_robustness import calculate_robustness

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
    # plt.plot(x_coordinates_1, y_coordinates_1)
    # plt.plot(x_coordinates_2, y_coordinates_2)
    # plt.plot(x_coordinates_3, y_coordinates_3)
    # plt.plot(x_coordinates_4, y_coordinates_4)

    return x_coordinates_1, y_coordinates_1, x_coordinates_2, y_coordinates_2, x_coordinates_3, y_coordinates_3, x_coordinates_4, y_coordinates_4


def branch_new_region_support(region_support: np.array, direction_of_branching:int, uniform: bool,branching_factor:int, rng) -> np.array:
    """Generate new region supports based on direction of branching and the branching factor. For now, the
    partitioning of space is uniformly done.

    Args:
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        direction_of_branching (int): Dimension or direction in which branching is to be done
        uniform (bool): Do we partition the region uniformly or randomly. If True, uniform partition takes place
        branching_factor (int): How many new region supports to be made from the previous region support

    Returns:
        np.array: New region support
    """

    dim_length = region_support[0][direction_of_branching][1] - region_support[0][direction_of_branching][0]
    if uniform:
        split_array = region_support[0][direction_of_branching][0] + (np.arange(branching_factor+1)/branching_factor) * dim_length
    else: 
        split_array = region_support[0][direction_of_branching][0] + np.sort(np.insert(rng.uniform(0,1,branching_factor-1), 0,[0,1])) * dim_length
        # print(split_array)
    
    
    new_bounds = []
    for i in range(branching_factor):
        temp = region_support.copy()
        temp[0,direction_of_branching,0] = split_array[i]
        temp[0,direction_of_branching,1] = split_array[i+1]
        # print(temp)
        # print("**************")
        new_bounds.append(temp[0,:,:])
    return np.array(new_bounds)


def pointsInSubRegion(samples_in: np.array, samples_out:np.array, regionBounds: list) -> list:
    """

    Args:
        samples_in (np.array): [description]
        regionBounds (list): [description]

    Returns:
        list: [description]
    """    
    regionSamples = []
    corresponding_robustenss = []
    for iterate,subregion in enumerate(regionBounds):
        boolArray = []
        for dimension in range(len(subregion)):
            subArray = samples_in[:, :, dimension]
            logical_subArray = np.logical_and(subArray >= subregion[dimension, 0],subArray <= subregion[dimension, 1])
            boolArray.append(np.squeeze(logical_subArray))
        corresponding_robustenss.append(samples_out[:,(np.all(boolArray, axis = 0))])
        regionSamples.append(samples_in[:,(np.all(boolArray, axis = 0)),:])
    return regionSamples, corresponding_robustenss


"""
Test the pointsInSubRegion function
"""


def testPointInSubRegion(regionSamples, regionBounds, subRegionBounds):
    x_coordinates_1 = [regionBounds[0,0,0], regionBounds[0,0,0]]
    y_coordinates_1 = [regionBounds[0,1,0], regionBounds[0,1,1]]

    x_coordinates_2 = [regionBounds[0,0,0], regionBounds[0,0,1]]
    y_coordinates_2 = [regionBounds[0,1,0], regionBounds[0,1,0]]

    x_coordinates_3 = [regionBounds[0,0,1], regionBounds[0,0,1]]
    y_coordinates_3 = [regionBounds[0,1,0], regionBounds[0,1,1]]

    x_coordinates_4 = [regionBounds[0,0,0], regionBounds[0,0,1]]
    y_coordinates_4 = [regionBounds[0,1,1], regionBounds[0,1,1]]

    
    # print(x_coordinates)
    # print(y_coordinates)
    plt.plot(x_coordinates_1, y_coordinates_1, color = 'red')
    plt.plot(x_coordinates_2, y_coordinates_2, color = 'red')
    plt.plot(x_coordinates_3, y_coordinates_3, color = 'red')
    plt.plot(x_coordinates_4, y_coordinates_4, color = 'red')

    listStyle = ['b--',  'y--', 'k--', 'g--']
    listStyle_marker = ['b.',  'y.', 'k.', 'g.']
    
    for iterate in range(len(subRegionBounds)):
        # print(subRegionBounds[iterate,:,:])
        # print("***********")

        x_coordinates_sub_r_1 = [subRegionBounds[iterate,0,0], subRegionBounds[iterate,0,0]]
        y_coordinates_sub_r_1 = [subRegionBounds[iterate,1,0], subRegionBounds[iterate,1,1]]

        x_coordinates_sub_r_2 = [subRegionBounds[iterate,0,0], subRegionBounds[iterate,0,1]]
        y_coordinates_sub_r_2 = [subRegionBounds[iterate,1,0], subRegionBounds[iterate,1,0]]

        x_coordinates_sub_r_3 = [subRegionBounds[iterate,0,1], subRegionBounds[iterate,0,1]]
        y_coordinates_sub_r_3 = [subRegionBounds[iterate,1,0], subRegionBounds[iterate,1,1]]

        x_coordinates_sub_r_4 = [subRegionBounds[iterate,0,0], subRegionBounds[iterate,0,1]]
        y_coordinates_sub_r_4 = [subRegionBounds[iterate,1,1], subRegionBounds[iterate,1,1]]

        plt.plot(x_coordinates_sub_r_1, y_coordinates_sub_r_1, listStyle[iterate])
        plt.plot(x_coordinates_sub_r_2, y_coordinates_sub_r_2, listStyle[iterate])
        plt.plot(x_coordinates_sub_r_3, y_coordinates_sub_r_3, listStyle[iterate])
        plt.plot(x_coordinates_sub_r_4, y_coordinates_sub_r_4, listStyle[iterate])
    
    for i, subregionPoints in enumerate(regionSamples):

        if subregionPoints.shape[1]!=0:
            # sr = (subregionPoints)
            # print(sr)
            # print(sr.shape)
            plt.plot(subregionPoints[0,:,0], subregionPoints[0,:,1], listStyle_marker[i])
    plt.show()


def assign_budgets(vol_probablity_distribution, continued_sampling_budget, rng):

    cumu_sum = np.cumsum(np.insert(vol_probablity_distribution, 0,0))
    # print("Cumulative_sum list = {}".format(cumu_sum))
    random_numbers = rng.uniform(0.0,1.0, continued_sampling_budget)
    n_cont_budget_distribution = []
    for iterate in range(len(cumu_sum)-1):
        bool_array = np.logical_and(random_numbers > cumu_sum[iterate], random_numbers <= cumu_sum[iterate+1])
        n_cont_budget_distribution.append(bool_array.sum())
    # print(n_cont_budget_distribution)
    return (n_cont_budget_distribution)


# region_support = np.array([[[-1.,1.], [-1.,1.]]])
# direction_of_branching = 1
# branchingFactor = 2
# problemDimension = 2

# partitions_region_support = branch_new_region_support(region_support, direction_of_branching, True, branchingFactor)

# print(partitions_region_support)
# print(partitions_region_support.shape)

# print(partitions_region_support[0].reshape((1,partitions_region_support[0].shape[0],partitions_region_support[0].shape[1])))

# number_of_samples = 100
# samples = lhs_sampling(number_of_samples, region_support, problemDimension)
# out = calculate_robustness(samples)
# rs, out = pointsInSubRegion(samples, out, partitions_region_support)

# testPointInSubRegion(rs, region_support, partitions_region_support)

