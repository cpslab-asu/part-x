import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
from scipy import stats
from .sampling import lhs_sampling
from kriging_gpr.interface.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from kriging_gpr.interface.OK_Rpredict import OK_Rpredict


def calculateQuantile(y_pred, sigma_st, alpha):
    """Min-Max Quantile Calculation

    Args:
        y_pred ([type]): Predicted Function Value for Sampled observation from the BO algorithm.
        sigma_st ([type]): Standard deviation of Fitted Gaussian Process corresponding to the region at query points.
        alpha ([type]): list of Significance Levels

    Returns:
        [type]: lower_quantile
        [type]: upper_quantile
    """

    term1 = np.array(y_pred)
    term2 = np.array(stats.norm.ppf(1 - (alpha / 2)) * (sigma_st))
    term2 = term2.reshape(term2.shape[0], 1)

    lower_quantile = term1 - term2
    upper_quantile = term1 + term2
    return lower_quantile, upper_quantile


#########################################MC-Estimates and CONFIDENCE INTERVAL###############################
def mc_Step(samples_in, samples_out, grid, region_support, regionDimensions, alpha, R, M, rng): #3
    """Function to run the MCStep algorithm in the paper. The idea is to take the exisitng samples
    and create a GP. Use this GP to predict the mean and the std_dev and calculate quantiles for
    region classification.

    Estimated Complexity = O(R*(M + len(alpha)))

    Args:
        samples_in (np.array): The exisitng input samples (points).
        samples_out (np.array): The output of the samples_in point (robustness values).
        grid (list): Array of RxM points. empty if region is already classified
        region_support (np.array): The bounds of a region.
        regionDimensions (int): Dimensionality of the region.
        alpha (list): List of confidence interval values (alpha) at which the quantiles are to calculated
        R (int): number of monte carlo iterations (refer to the paper)
        M (int): number of evaluation per iteration (refer to the paper).

    Returns:
        list: min and max quantiles
    """

    
    minQuantile = np.zeros((R, len(alpha)))
    maxQuantile = np.zeros((R, len(alpha)))
    
    grid_list = grid.tolist()[0]
    reshaped_grid = [[grid_list[i:i + M]] for i in range(0, len(grid_list), M)]
    # print(np.array(reshaped_grid).shape)
    # print(np.array(reshaped_grid).shape)
    for iterate in range(R):
        X = samples_in[0]
        Y = np.transpose(samples_out)
        model = OK_Rmodel_kd_nugget(X, Y, 0, 2)
        


        samples = reshaped_grid[iterate]
        # print((np.array(samples)[0]))
            # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py
        y_pred, sigma_st = OK_Rpredict(model, np.array(samples)[0], 0, Y)
        for alpha_iter in range(len(alpha)):
            minq, maxq = calculateQuantile(y_pred, sigma_st, alpha[alpha_iter])
            minQuantile[iterate, alpha_iter] = min(minq)
            maxQuantile[iterate, alpha_iter] = max(maxq)
            # print(minq)
    return minQuantile, maxQuantile


def estimateMC(lower_quantile: list, upper_quantile: list):
    """calculate mean and variance from lower and upper quantiles

    Args:
        lower_quantile (list): lower quantile values
        upper_quantile (list): upper quantile values

    Returns:
        list: minimum quantile mean, minimum quantile variance, maximum quantile mean, maximum quantile variance
    """
    R = lower_quantile.shape[0]
    # print(minQuantile.shape)
    mcEstimate_minimum_mean = (np.mean(lower_quantile, 0))
    mcEstimate_maximum_mean = (np.mean(upper_quantile, 0))

    mcEstimate_minimum_variance = (np.var(lower_quantile, 0)) / R
    mcEstimate_maximum_variance = (np.var(upper_quantile, 0)) / R

    # print("Min Mean = {}\tMax Mean = {}".format(mcEstimate_minimum_mean, mcEstimate_maximum_mean))
    # print("Min Variance = {}\tMax variance = {}".format(mcEstimate_minimum_variance, mcEstimate_maximum_variance))
    return mcEstimate_minimum_mean, mcEstimate_minimum_variance, mcEstimate_maximum_mean, mcEstimate_maximum_variance



def estimate_quantiles(samples_in: np.array, samples_out: np.array, grid:list, region_support:np.array, regionDimensions:int, alpha:list, R:int, M:int, rng)->list:
    """Main driver function for estimating the lower and upper bounds from samples

    Args:
        samples_in (np.array): The exisitng input samples (points).
        samples_out (np.array): The output of the samples_in point (robustness values).
        grid (list):RxM points. empty if region is already classified
        region_support (np.array): The bounds of a region.
        regionDimensions (int): Dimensionality of the region.
        alpha (list): List of confidence interval values (alpha) at which the quantiles are to calculated
        R (int): number of monte carlo iterations (refer to the paper)
        M (int): number of evaluation per iteration (refer to the paper).

    Returns:
        list: lower and upper bounds
    """
    lower_quantile, upper_quantile = mc_Step(samples_in, samples_out, grid, region_support, regionDimensions, alpha, R, M, rng)
    mcEstimate_minimum_mean, mcEstimate_minimum_variance, mcEstimate_maximum_mean, mcEstimate_maximum_variance = estimateMC(lower_quantile, upper_quantile)
    # print(mcEstimate_minimum_mean)
    # print(mcEstimate_minimum_variance)
    # print(mcEstimate_maximum_mean)
    # print(mcEstimate_maximum_variance)
    lower_bound = []
    upper_bound = []
    for alpha_iter in range(len(alpha)):
        lower_bound.append(mcEstimate_minimum_mean[alpha_iter] - ((stats.norm.ppf(1 - (alpha[alpha_iter] / 2))) * mcEstimate_minimum_variance[alpha_iter]))
        upper_bound.append(mcEstimate_maximum_mean[alpha_iter] + ((stats.norm.ppf(1 - (alpha[alpha_iter] / 2))) * mcEstimate_maximum_variance[alpha_iter]))
    # print(lower_bound, upper_bound)
    return lower_bound, upper_bound

#####################################################Test################################################################################################

#########################################Test######################################################
# region_support = np.array([[[-1, 1], [-1, 1]]])
# regionDimension = 2
# numberOfSamples = 100

# samples_in = lhs_sampling(numberOfSamples, region_support, regionDimension)
# samples_out = calculate_robustness(samples_in)

# alpha = [0.5, 0.95, 0.99]

# # a,b=mc_Step(samples_in, samples_out, region_support, regionDimension, alpha,2,5)
# # print(a)
# # print(b)



# # q_min_m, q_min_v, q_max_m, q_max_v = estimateMC(a, b)
# # L_Bound = q_min_m - ((stats.norm.ppf(1 - (0.95 / 2))) * q_min_v)
# # U_Bound = q_max_m + ((stats.norm.ppf(1 - (0.95 / 2))) * q_max_v)
# # print("L_Bound = {}\tU_Bound = {}".format(L_Bound, U_Bound))

# R = 100
# M = 1000
# lower_bound, upper_bound = estimate_quantiles(samples_in, samples_out, region_support, regionDimension, alpha,R, M)

# print(lower_bound)
# print("************")
# print(upper_bound)

# print("L_Bound = {}\tU_Bound = {}".format(lower_bound, upper_bound))