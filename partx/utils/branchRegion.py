import numpy as np

def branch_region(region_support: np.array, direction_of_branching:int, uniform: bool,branching_factor:int, rng) -> np.array:
    """Generate new region supports based on direction of branching and the branching factor. For now, the
    partitioning of space is uniformly done.

    Args:
        region_support: The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        direction_of_branching: Dimension or direction in which branching is to be done
        uniform: Do we partition the region uniformly or randomly. If True, uniform partition takes place
        branching_factor: How many new region supports to be made from the previous region support

    Returns:
        np.array: New region support
    """

    dim_length = region_support[direction_of_branching][1] - region_support[direction_of_branching][0]
    if uniform:
        split_array = region_support[direction_of_branching][0] + (np.arange(branching_factor+1)/branching_factor) * dim_length
    else: 
        split_array = region_support[direction_of_branching][0] + np.sort(np.insert(rng.uniform(0,1,branching_factor-1), 0,[0,1])) * dim_length
        # print(split_array)
    
    
    new_bounds = []
    for i in range(branching_factor):
        temp = region_support.copy()
        temp[direction_of_branching,0] = split_array[i]
        temp[direction_of_branching,1] = split_array[i+1]
        
        new_bounds.append(temp)
    return new_bounds
