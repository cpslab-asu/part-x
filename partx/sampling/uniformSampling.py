import numpy as np
import numpy.typing as npt

class OOBError(ValueError): pass

# def uniform_sampling(
#     num_samples: int, region_support: npt.NDArray, tf_dim: int, rng
# ) -> np.array:
#     """Sample *num_samples* points within the *region_support* which has a dimension as mentioned below.

#     Args:
#         num_samples: Number of points to sample within the region bounds.
#         region_support: The bounds of the region within which the sampling is to be done.
#                                     Region Bounds is N x O where;
#                                         N = tf_dim (Dimensionality of the test function);
#                                         O = Lower and Upper bound. Should be of length 2;
#         tf_dim: The dimensionality of the region. (Dimensionality of the test function)

#     Returns:
#         np.array: 3d array with samples between the bounds.
#                     Size of the array will be N x O
#                         N = num_samples
#                         O = tf_dim (Dimensionality of the test function)
#     """

#     if region_support.shape[0] != tf_dim:
#         raise ValueError(f"Region Support has wrong dimensions. Expected {tf_dim}, received {region_support.shape[0]}")
#     if region_support.shape[1] != 2:
#         raise ValueError("Region Support matrix must be MxNx2")
    
#     if not np.alltrue(region_support[:,1]-region_support[:,0] >= 0):
#         raise ValueError("Region Support Z-pairs must be in increasing order")

#     raw_samples = np.apply_along_axis(
#         lambda bounds: rng.uniform(bounds[0], bounds[1], num_samples),
#         1,
#         region_support,
#     )

#     samples = []
#     for sample in raw_samples:
#         samples.append(sample)

#     return np.array(samples).T


def uniform_sampling(
    num_samples: int, region_support: npt.NDArray, tf_dim: int, oracle_info, rng
) -> np.array:
    """Sample *num_samples* points within the *region_support* which has a dimension as mentioned below,
    satisfying the constraints defined by the *oracle_func*.

    Args:
        num_samples: Number of points to sample within the region bounds.
        region_support: The bounds of the region within which the sampling is to be done.
                        Region Bounds is N x O where;
                            N = tf_dim (Dimensionality of the test function);
                            O = Lower and Upper bound. Should be of length 2;
        tf_dim: The dimensionality of the region. (Dimensionality of the test function)
        rng: Random number generator object.
        oracle_func: The oracle function that encodes the constraints.

    Returns:
        np.array: 3d array with samples between the bounds.
                  Size of the array will be N x O
                      N = num_samples
                      O = tf_dim (Dimensionality of the test function)
    """

    if region_support.shape[0] != tf_dim:
        raise ValueError(f"Region Support has wrong dimensions. Expected {tf_dim}, received {region_support.shape[0]}")
    if region_support.shape[1] != 2:
        raise ValueError("Region Support matrix must be MxNx2")
    
    if not np.alltrue(region_support[:, 1] - region_support[:, 0] >= 0):
        raise ValueError("Region Support Z-pairs must be in increasing order")

    samples = []
    n_tries = oracle_info.n_tries_randomsampling
    while len(samples) < num_samples and n_tries > 0:
        point = np.array([rng.uniform(bounds[0], bounds[1]) for bounds in region_support])
        if oracle_info(point).sat:
            samples.append(point)
            n_tries = oracle_info.n_tries_randomsampling
        else:
            n_tries -= 1
    
    if n_tries == 0 and len(samples)!=num_samples:
        raise OOBError(f"Could not perform random sampling, {oracle_info.n_tries_randomsampling} trials exhausted. Please adjust the constraints or increase the budget for n_trials_randomsampling")
    return np.array(samples)