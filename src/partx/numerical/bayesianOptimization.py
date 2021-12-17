#!/usr/bin/env python3
import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from numpy import argmax
from scipy.stats import norm
from pathos.multiprocessing import ProcessingPool as Pool
from .calculate_robustness import calculate_robustness
from .sampling import lhs_sampling
from .sampling import uniform_sampling

from kriging_gpr.interface.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from kriging_gpr.interface.OK_Rpredict import OK_Rpredict

from scipy.optimize import minimize, dual_annealing
from scipy.optimize import Bounds

def exists_in(X, sample, dimension):
    
    for x_ in X:
        if sum(sample == x_) == dimension:
            return True
        else:
            return False

def surrogate(model, X:np.array):
    """Surrogate Model function

    Args:
        model ([type]): Gaussian process model
        X (np.array): Input points

    Returns:
        [type]: predicted values of points using gaussian process model
    """
	# catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        # return model.predict(X, return_std=True)
        return OK_Rpredict(model, X, 0)


def acquisition(X: np.array, Xsamples: np.array, model):
    """Acquisition function

    Args:
        X (np.array): sample points 
        Xsamples (np.array): randomly sampled points for calculating surrogate model mean and std
        model ([type]): Gaussian process model

    Returns:
        [type]: Sample probabiility of each sample points
    """
    # if (Xsamples.shape).shape == 1:
    Xsamples = Xsamples.reshape(1,Xsamples.shape[0])
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X) 
    curr_best = np.min(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    
    mu = mu[0,0]
    std = std[0,0]
    ei_0 = []
    # print(std)
    # print("******************************")
    pred_var = std
    if pred_var > 0:
        
        var_1 = curr_best-mu
        var_2 = var_1 / pred_var
        
        
        ei = ((var_1 * norm.cdf(var_2,loc=0,scale=1)) + (pred_var * norm.pdf(var_2,loc=0,scale=1)))
    else:
        ei = 0.0
    return ei

def opt_acquisition(X: np.array, model, sbo:list ,test_function_dimension:int, region_support: np.array, rng) -> np.array:
    """Get the sample points

    Args:
        X (np.array): sample points 
        y (np.array): corresponding robustness values
        model ([type]): the GP models 
        sbo (list): sample points to construct the robustness values
        test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;

    Returns:
        [np.array]: the new sample points by BO
        [np.array]: sbo - new samples for resuse
    """
    # print("*************")
    # print("Length before removing {}".format(sbo.shape))

    region_support = np.array(region_support.reshape((1,region_support.shape[0],region_support.shape[1])))
    lower_bound_theta = np.ndarray.flatten(region_support[0,:,0])
    upper_bound_theta = np.ndarray.flatten(region_support[0,:,1])

    bnds =  Bounds(lower_bound_theta, upper_bound_theta)
    fun = lambda x_: -1*acquisition(X,x_,model)
    # random_sample = uniform_sampling(1, region_support, test_function_dimension, rng)
    
    # options = {'maxiter':1e10}
    # params = minimize(fun, np.ndarray.flatten(random_sample[:,0,:]), method = 'L-BFGS-B', bounds = bnds, options = options)
    params_2 = dual_annealing(fun, bounds = list(zip(lower_bound_theta, upper_bound_theta)), no_local_search = False)
    # print(params)
    # print("DA result below")
    # print(params_2)
    min_bo = params_2.x

    while exists_in(X, min_bo, test_function_dimension):
        # print("Rerunning Optimizer to avoid duplicate point")
        random_sample = uniform_sampling(1, region_support, test_function_dimension, rng)
        params_2 = dual_annealing(fun, bounds = list(zip(lower_bound_theta, upper_bound_theta)), no_local_search = False, x0 = np.ndarray.flatten(random_sample[:,0,:]))
    
        min_bo = params_2.x

    new_sbo = np.delete(sbo, 0, axis = 1)
    return np.array(min_bo), new_sbo


# def opt_acquisition(X: np.array, y: np.array, model, sbo:list ,test_function_dimension:int, region_support: np.array, rng) -> np.array:
#     """Get the sample points

#     Args:
#         X (np.array): sample points 
#         y (np.array): corresponding robustness values
#         model ([type]): the GP models 
#         sbo (list): sample points to construct the robustness values
#         test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
#         region_support (np.array): The bounds of the region within which the sampling is to be done.
#                                     Region Bounds is M x N x O where;
#                                         M = number of regions;
#                                         N = test_function_dimension (Dimensionality of the test function);
#                                         O = Lower and Upper bound. Should be of length 2;

#     Returns:
#         [np.array]: the new sample points by BO
#         [np.array]: sbo - new samples for resuse
#     """
#     # print("*************")
#     # print("Length before removing {}".format(sbo.shape))
#     region_support = np.array(region_support.reshape((1,region_support.shape[0],region_support.shape[1])))
#     scores = acquisition(X, y, sbo, model)
#     ix = argmax(scores)
#     min_bo = sbo[0,ix,:]
#     new_sbo = np.delete(sbo, ix, axis = 1)
#     # print("Length after removing {}".format(new_sbo.shape))
#     # print("*************")
#     return np.array(min_bo), new_sbo
    



def bayesian_optimization(test_function, samples_in: np.array, corresponding_robustness: np.array, number_of_samples_to_generate: list, test_function_dimension:int, region_support:list, random_points_for_gp: list, rng) -> list:
    """Sample using Bayesian Optimization
    https://machinelearningmastery.com/what-is-bayesian-optimization/

    Args:
        samples_in (np.array): Sample points
        corresponding_robustness (np.array): Robustness values
        number_of_samples_to_generate (list): Number of samples to generate using BO
        test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        random_points_for_gp (list): Random samlpes for SBO

    Returns:
        list: Old and new samples (np.array of shape M x N x O). Length of list is number of regions provided in samples_in
                        M = number of regions
                        N = number_of_samples
                        O = test_function_dimension (Dimensionality of the test function) )
        list: corresponding robustness
        list: samples from acquisition function for reuse in classification
    """

    samples_in_new = []
    corresponding_robustness_new = []
    sbo = random_points_for_gp
    for i in range(samples_in.shape[0]):
        X = samples_in[i,:,:]
        Y = corresponding_robustness[i,:].reshape((corresponding_robustness.shape[1],1))
        
        for j in range(number_of_samples_to_generate[i]):
            model = OK_Rmodel_kd_nugget(X, Y, 0, 2, 16)
            min_bo, sbo = opt_acquisition(X, model, sbo, test_function_dimension, region_support[i,:,:], rng)
            actual = calculate_robustness(np.array(min_bo), test_function)
            # print("*****************************")
            # print(min_bo)
            # print(actual)
            # print("*****************************")
            X = np.vstack((X, np.array(min_bo)))
            Y = np.vstack((Y, np.array(actual)))
        samples_in_new.append(np.expand_dims(X, axis = 0))
        corresponding_robustness_new.append(np.transpose(Y))
    return samples_in_new, corresponding_robustness_new


# rng = np.random.default_rng(seed)
# region_support = np.array([[[-1, 1], [-1, 1]]])
# test_function_dimension = 2
# number_of_samples = 20

# x = lhs_sampling(number_of_samples, region_support, test_function_dimension, rng)
# y = calculate_robustness(x)

# x_new, y_new, s = bayesian_optimization(x, y, [10], test_function_dimension, region_support, 10, rng)
# return x_new, y_new,s



# def run_par(data):
#     num_samples, BO_samples, s = data
#     rng = np.random.default_rng(s)
#     region_support = np.array([[[-1, 1], [-1, 1]]])
#     test_function_dimension = 2
#     number_of_samples = num_samples

#     x = lhs_sampling(number_of_samples, region_support, test_function_dimension, rng)
#     y = calculate_robustness(x)

#     x_new, y_new, s = bayesian_optimization(x, y, BO_samples, test_function_dimension, region_support, 10, rng)
#     print(test_function.callCount)
#     return [test_function.callCount]

# inputs = []
# start_seed = 1
# a = [10,10,10,10]
# b = [[20],[21],[19],[22]]
# for q in range(4):
#     s =  start_seed + q
#     data = (a[q], b[q], s)
#     inputs.append(data)

# pool = Pool()
# results = list(pool.map(run_par, inputs))

# print(results)
