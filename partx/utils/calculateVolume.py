
import numpy as np
import numpy.typing as npt

def calculate_volume(region_support: npt.NDArray) -> list:
    """Calculate volume of a hypercube. 

    Args:
        region_support: The bounds of the region within which the sampling is to be done.
                                    Region Bounds is N x O where;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;

    Returns:
        float: volume. List of length = number of regions
    """
    return np.prod(region_support[:,1]-region_support[:,0], axis = 0)
