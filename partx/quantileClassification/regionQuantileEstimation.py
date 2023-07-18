import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
from scipy import stats
from ..gprInterface import GPR
from ..sampling import uniform_sampling, lhs_sampling

def calculate_quantile(y_pred, sigma_st, alpha):
    """Min-Max Quantile Calculation

    Args:
        y_pred: Predicted Function Value for Sampled observation from the BO algorithm.
        sigma_st: Standard deviation of Fitted Gaussian Process corresponding to the region at query points.
        alpha: list of Significance Levels

    Returns:
        [type]: lower_quantile
        [type]: upper_quantile
    """

    term1 = y_pred
    term2 = stats.norm.ppf(1 - (alpha / 2)) * sigma_st

    lower_quantile = term1 + term2
    upper_quantile = term1 - term2
    # lower_quantile = term1 - term2
    # upper_quantile = term1 + term2
    return lower_quantile, upper_quantile


#########################################MC-Estimates and CONFIDENCE INTERVAL###############################
def mc_step(x_train, y_train, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type): 
    """Function to run the MCStep algorithm in the paper. The idea is to take the exisitng samples
    and create a GP. Use this GP to predict the mean and the std_dev and calculate quantiles for
    region classification.

    Estimated Complexity = O(R*(M + len(alpha)))

    Args:
        samples_in: The exisitng input samples (points).
        samples_out: The output of the samples_in point (robustness values).
        grid: Array of RxM points. empty if region is already classified
        region_support: The bounds of a region.
        regionDimensions: Dimensionality of the region.
        alpha: List of confidence interval values (alpha) at which the quantiles are to calculated
        R: number of monte carlo iterations (refer to the paper)
        M: number of evaluation per iteration (refer to the paper).

    Returns:
        list: min and max quantiles
    """

    
    minQuantile = np.zeros((R, 1))
    maxQuantile = np.zeros((R, 1))
    
    for iterate in range(R):
        # model = OK_Rmodel_kd_nugget(X, Y, 0, 2, gpr_params)
        model = GPR(gpr_model)
        model.fit(x_train, y_train)
        
        if sampling_type == "lhs_sampling":
            samples = lhs_sampling(M, region_support, tf_dim, oracle_info, rng)
        elif sampling_type == "uniform_sampling":
            samples = uniform_sampling(M, region_support, tf_dim, oracle_info, rng)
        else:
            raise ValueError(f"{sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        y_pred, sigma_st = model.predict(samples)
        
        
        lower_q, upper_q = calculate_quantile(y_pred, sigma_st, alpha)
        
        minQuantile[iterate, 0] = max(lower_q)
        maxQuantile[iterate, 0] = min(upper_q)
        # minQuantile[iterate, 0] = min(lower_q)
        # maxQuantile[iterate, 0] = max(upper_q)
            
    return minQuantile, maxQuantile


def estimate_mc(lower_quantile: list, upper_quantile: list):
    """calculate mean and variance from lower and upper quantiles

    Args:
        lower_quantile: lower quantile values
        upper_quantile: upper quantile values

    Returns:
        list: minimum quantile mean, minimum quantile variance, maximum quantile mean, maximum quantile variance
    """
    R = lower_quantile.shape[0]
    
    # print(minQuantile.shape)
    mcEstimate_minimum_mean = (np.mean(lower_quantile, 0))
    mcEstimate_maximum_mean = (np.mean(upper_quantile, 0))

    mcEstimate_minimum_variance = (np.var(lower_quantile, 0)) / R
    mcEstimate_maximum_variance = (np.var(upper_quantile, 0)) / R
    
    return mcEstimate_minimum_mean, mcEstimate_minimum_variance, mcEstimate_maximum_mean, mcEstimate_maximum_variance



def estimate_quantiles(x_train: np.array, y_train: np.array, region_support:np.array, tf_dim:int, alpha:list, R:int, M:int, gpr_model, oracle_info, rng, sampling_type = "lhs_sampling")->list:
    """Main driver function for estimating the lower and upper bounds from samples

    Args:
        samples_in: The exisitng input samples (points).
        samples_out: The output of the samples_in point (robustness values).
        grid: RxM points. empty if region is already classified
        region_support: The bounds of a region.
        regionDimensions: Dimensionality of the region.
        alpha: List of confidence interval values (alpha) at which the quantiles are to calculated
        R: number of monte carlo iterations (refer to the paper)
        M: number of evaluation per iteration (refer to the paper).

    Returns:
        list: lower and upper bounds
    """
    if x_train.shape[0] > 0 and x_train.shape[0] == y_train.shape[0]:
        min_quantile, max_quantile = mc_step(x_train, y_train, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type)
        mcEstimate_minimum_mean, mcEstimate_minimum_variance, mcEstimate_maximum_mean, mcEstimate_maximum_variance = estimate_mc(min_quantile, max_quantile)
        # print(mcEstimate_minimum_mean)
        # print(mcEstimate_minimum_variance)
        # print(mcEstimate_maximum_mean)
        # print(mcEstimate_maximum_variance)
        # lower_bound = []
        # upper_bound = []
        
        min_delta_quantile = (mcEstimate_minimum_mean + ((stats.norm.ppf(1 - (alpha / 2))) * mcEstimate_minimum_variance))
        max_delta_quantile = (mcEstimate_maximum_mean - ((stats.norm.ppf(1 - (alpha / 2))) * mcEstimate_maximum_variance))
    else:
        min_delta_quantile = np.array([None])
        max_delta_quantile = np.array([None])
    # print(lower_bound, upper_bound)
    return min_delta_quantile[0], max_delta_quantile[0]
