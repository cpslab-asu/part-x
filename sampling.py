
import numpy as np
from numpy.core.fromnumeric import shape
from testFunction import test_function
from calculate_robustness import calculate_robustness


def uniformSampling(number_of_samples: int, region_support: np.array, test_function_dimension: int) -> np.array:
    """Sample *number_of_samples* points within the *region_support* which has a dimension as mentioned below.

    Args:
        number_of_samples (int): Number of points to sample within the region bounds.
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)

    Returns:
        np.array: 3d array with samples between the bounds.
                    Size of the array will be M x N x O
                        M = number of regions
                        N = number_of_samples
                        O = test_function_dimension (Dimensionality of the test function)
    """
    
    assert region_support.shape[1] == test_function_dimension, 'sub_r matrix must be 3-dimensional'
    assert region_support.shape[2] == 2, 'sub_r matrix must be MxNx2'
    assert np.apply_along_axis(lambda bounds: bounds[1] > bounds[0], 2, region_support).all(), 'sub_r Z-pairs must be in increasing order'

    raw_samples = np.apply_along_axis(
            lambda bounds: 
                np.random.uniform(bounds[0], bounds[1], number_of_samples), 
            2, region_support)

    samples = []
    for sample in raw_samples:
        samples.append(np.transpose(sample))

    return np.array(samples)



# region_support = np.array([[[-1, 1], [-1, 1]]])
# test_function_dimension = 2
# number_of_samples = 3

# samples_in = uniformSampling(number_of_samples, region_support, test_function_dimension)
# # print(samples_in.shape)



# samples_out = calculate_robustness(samples_in)
# print(samples_in.shape)
# b = np.array([[[]]])
# print(b.shape)
# a = np.append(samples_in,samples_in, axis=1)

# print(a.shape)
# print(test_function.callCount)
