import numpy as np
import numpy.typing as npt

def uniform_sampling(
    num_samples: int, region_support: npt.NDArray, tf_dim: int, rng
) -> np.array:
    """Sample *num_samples* points within the *region_support* which has a dimension as mentioned below.

    Args:
        num_samples: Number of points to sample within the region bounds.
        region_support: The bounds of the region within which the sampling is to be done.
                                    Region Bounds is N x O where;
                                        N = tf_dim (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        tf_dim: The dimensionality of the region. (Dimensionality of the test function)

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
    
    if not np.alltrue(region_support[:,1]-region_support[:,0] >= 0):
        raise ValueError("Region Support Z-pairs must be in increasing order")

    raw_samples = np.apply_along_axis(
        lambda bounds: rng.uniform(bounds[0], bounds[1], num_samples),
        1,
        region_support,
    )

    samples = []
    for sample in raw_samples:
        samples.append(sample)

    return np.array(samples).T
