# import numpy as np
# import numpy.typing as npt
# from scipy.stats import qmc


# def lhs_sampling(
#     num_samples: int,
#     region_support: npt.NDArray,
#     tf_dim: int,
#     rng,
# ) -> np.array:
#     """Latin Hypercube Sampling: Sample *num_samples* points within the *region_support* which has a dimension as mentioned below.

#     Args:
#         num_samples: Number of points to sample within the region bounds.
#         region_support: The bounds of the region within which the sampling is to be done.
#                                     Region Bounds is N x O where;
#                                         N = tf_dim (Dimensionality of the test function);
#                                         O = Lower and Upper bound. Should be of length 2;
#         tf_dim: The dimensionality of the region. (Dimensionality of the test function)

#     Returns:
#         np.array: 3d array with samples between the bounds.
#                     Size of the array will be M x N x O
#                         N = num_samples
#                         O = tf_dim (Dimensionality of the test function)
#     """

#     if region_support.shape[0] != tf_dim:
#         raise ValueError(f"Region Support has wrong dimensions. Expected {tf_dim}, received {region_support.shape[0]}")
#     if region_support.shape[1] != 2:
#         raise ValueError("Region Support matrix must be MxNx2")
    
#     if not np.alltrue(region_support[:,1]-region_support[:,0] >= 0):
#         raise ValueError("Region Support Z-pairs must be in increasing order")

#     sampler = qmc.LatinHypercube(d=tf_dim, seed=rng)
#     samples = sampler.random(n=num_samples)
#     lb = region_support[:,0]
#     ub = region_support[:,1]
#     # print(lb, ub)
#     scaled_samples = qmc.scale(samples, lb, ub)

#     return np.array(scaled_samples)
import numpy as np
import numpy.typing as npt
from scipy.stats import qmc
from .uniformSampling import OOBError

def lhs_sampling(
    num_samples: int,
    region_support: npt.NDArray,
    tf_dim: int,
    oracle_info,
    rng,
    ) -> np.array:
    """Latin Hypercube Sampling: Sample *num_samples* points within the *region_support* while respecting the constraints defined by the *oracle_func*.

    Args:
        num_samples: Number of points to sample within the region bounds.
        region_support: The bounds of the region within which the sampling is to be done.
                        Region Bounds is N x O where;
                            N = tf_dim (Dimensionality of the test function);
                            O = Lower and Upper bound. Should be of length 2;
        tf_dim: The dimensionality of the region. (Dimensionality of the test function)
        rng: Random number generator object.
        oracle_info: An object containing the oracle function and n_tries_randomsampling.

    Returns:
        np.array: 3d array with samples between the bounds.
                  Size of the array will be M x N x O
                      N = num_samples
                      O = tf_dim (Dimensionality of the test function)
    """

    if region_support.shape[0] != tf_dim:
        raise ValueError(f"Region Support has wrong dimensions. Expected {tf_dim}, received {region_support.shape[0]}")
    if region_support.shape[1] != 2:
        raise ValueError("Region Support matrix must be MxNx2")

    lb = region_support[:, 0]
    ub = region_support[:, 1]

    samples = []
    n_tries = oracle_info.n_tries_randomsampling

    while len(samples) < num_samples and n_tries > 0:
        
        lhs_samples = qmc.LatinHypercube(d=tf_dim, seed=rng).random(n=num_samples - len(samples))
        scaled_samples = qmc.scale(lhs_samples, lb, ub)

        for point in scaled_samples:
            # print(oracle_info(point).val)
            if oracle_info(point).sat:
                samples.append(point)
                # n_tries = oracle_info.n_tries_randomsampling
        if num_samples - len(samples) > 0:
            n_tries -=1
    
    

    
    if n_tries <= 0 and len(samples)!=num_samples:
        raise OOBError(f"Could not perform random sampling, {oracle_info.n_tries_randomsampling} trials exhausted. Please adjust the constraints or increase the budget for n_trials_randomsampling")
    return np.array(samples)
