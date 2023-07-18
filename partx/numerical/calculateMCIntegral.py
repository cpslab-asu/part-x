from ..gprInterface import GPR
from ..sampling import lhs_sampling, uniform_sampling
from ..utils import calculate_volume
import numpy as np
from scipy import stats


def calculate_mc_integral(x_train, y_train, region_support, tf_dim, R, M, gpr_model, oracle_info, rng, sampling_type):
    model = GPR(gpr_model)
    model.fit(x_train, y_train)

    cdf_all_sum = 0
    
    for _ in range(R):
        if sampling_type == "lhs_sampling":
            samples = lhs_sampling(M, region_support, tf_dim, oracle_info, rng)
        elif sampling_type == "uniform_sampling":
            samples = uniform_sampling(M, region_support, tf_dim, oracle_info, rng)
        else:
            raise ValueError(f"{sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        y_pred, pred_sigma = model.predict(samples)

        
        cdf_all_sum += np.sum(stats.norm.cdf(0., y_pred, pred_sigma))

    return (cdf_all_sum/(R*M)) * calculate_volume(region_support)
    # return calculate_volume(region_support)
