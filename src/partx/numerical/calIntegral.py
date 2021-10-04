from sklearn.gaussian_process import GaussianProcessRegressor
from .sampling import lhs_sampling
import numpy as np
from scipy import stats
# from calculate_robustness import calculate_robustness


def calculate_mc_integral(samples_in, samples_out, region_support, region_dimension, R, M, rng):
    X = samples_in[0]
    Y = np.transpose(samples_out)
    model = GaussianProcessRegressor()
    model.fit(X, Y)

    cdf_all = []
    for r in range(R):
        samples = lhs_sampling(M, region_support, region_dimension, rng)
        y_pred, sigma_st = model.predict(samples[0], return_std=True)
        for x in range(M):
            # print(stats.norm.cdf(0,y_pred[x],sigma_st[x]))
            cdf_all.extend((stats.norm.cdf(0.,y_pred[x],sigma_st[x])))
    # print(np.array(cdf_all).shape)
    return np.sum(cdf_all)/(R*M)




# region_support = np.array([[[-1, 1], [-1, 1]]])
# region_dimension = 2
# R = 10
# M = 100
# number_of_samples = 10
# samples_in = lhs_sampling(number_of_samples, region_support, region_dimension)
# samples_out = calculate_robustness(samples_in)

# x = calculate_mc_integral(samples_in, samples_out, region_support, region_dimension, R, M)
# print(x)