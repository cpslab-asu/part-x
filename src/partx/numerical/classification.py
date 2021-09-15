import numpy as np

def calculate_volume(region_support: np.array) -> list:
    """Calculate volume of a hypercube. 

    Args:
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;

    Returns:
        float: volume. List of length = number of regions
    """
    return np.prod(region_support[:,:,1]-region_support[:,:,0], axis = 1)
    
def classification(region_support: np.array, region_class:chr, min_volume:float, lower_bound:list, upper_bound:list)->chr:
    """Function classifies the region based on the its Quantile estimates
        TODO: modify for multiple region support

    Args:
        region_support (np.array): The bounds of a region.
        region_class (chr): The class of each region in previous iteration
        min_volume (float): Minimum Volume threshold for Classification
        lower_bound (list): lower bound of quantile estimates
        upper_bound (list): upper bound of quantile estimates


    Returns:
        chr: list of regions with corresponding class (Unidentified,Plus,Minus,RPlus,RMinus,Rem)
    """
    volume = calculate_volume(region_support)
    # dim_length = regionBounds2[:,:,1]-regionBounds2[:,:,0]
    # print(max_lengths)
    # print(max_lengths[:,direction_of_branch])
    for i in range(len(region_support)):
        if volume[i] <= min_volume:
            region_class = 'u'
        elif region_class == "+":
            if lower_bound[0] <= 0:
                region_class = 'r+'
            else:
                region_class = '+'
        elif region_class == "-":
            if upper_bound[0] >= 0:
                region_class = 'r-'
            else:
                region_class = "-"
        elif region_class == 'r':
            if upper_bound[0] < 0:
                region_class = "-"
            elif lower_bound[0] > 0:
                region_class = "+"
            else:
                region_class = 'r'
    return region_class


# regionBounds= np.array([[[-1, 1], [-1,1]]])

# sub_regionBounds_2 = np.array([[[-1, 1], [0,1]],[[0, 1], [-1,1]]])

# dimensionality = 2
# delta = 0.001

# volume_region_support = calculate_volume(regionBounds)
# min_volume = delta ** dimensionality * volume_region_support

# regionBounds2= np.array([[[-1, 1], [0, 5]],[[-3, 4], [-6, 7]]])
# lower_bound = np.array([[-1.23],[-3.45]])
# upper_bound = np.array([[3.24],[-2.45]])
# volume = np.array([3,4])
# delta=0.001
# print(len(regionBounds2))
# region_class = np.array(['r','r'])
# print(calculate_volume(regionBounds2))
# direction_of_branch = 1
# print(classification(regionBounds2, region_class, delta, lower_bound, upper_bound, direction_of_branch))