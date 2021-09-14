#!/usr/bin/env python3
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from numpy import argmax
from scipy.stats import norm
from pathos.multiprocessing import ProcessingPool as Pool
from calculate_robustness import calculate_robustness
from sampling import uniformSampling
# from testFunction import test_function

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
        return model.predict(X, return_std=True)


def acquisition(X: np.array, Xsamples: np.array, model):
    """Acquisition function

    Args:
        X (np.array): sample points 
        Xsamples (np.array): randomly sampled points for calculating surrogate model mean and std
        model ([type]): Gaussian process model

    Returns:
        [type]: Sample probabiility of each sample points
    """

    
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = min(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples[0,:,:])
    mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs



def opt_acquisition(X: np.array, y: np.array, model, num_points_to_construct_gp:int ,test_function_dimension:int, region_support: np.array, rng) -> np.array:
    """Get the sample points

    Args:
        X (np.array): sample points 
        y (np.array): corresponding robustness values
        model ([type]): the GP models 
        num_points_to_construct_gp (int): number of sample points to construct the robustness values
        test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;

    Returns:
        [np.array]: the new sample points by BO
        [np.array]: random uniform samples for reuse later
    """

    region_support = np.array(region_support.reshape((1,region_support.shape[0],region_support.shape[1])))
    sbo = uniformSampling(num_points_to_construct_gp, region_support, test_function_dimension, rng)
    scores = acquisition(X, sbo, model)
    ix = argmax(scores)
    min_bo = sbo[0,ix,:]
    return np.array(min_bo), sbo
    



def bayesian_optimization(test_function, samples_in: np.array, corresponding_robustness: np.array, number_of_samples_to_generate: list, test_function_dimension:int, region_support:list, num_points_to_construct_gp, rng) -> list:
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
        num_points_to_construct_gp ([int], optional): Number of points to construct GPs in BO. Defaults to 100:int.

    Returns:
        list: Old and new samples (np.array of shape M x N x O). Length of list is number of regions provided in samples_in
                        M = number of regions
                        N = number_of_samples
                        O = test_function_dimension (Dimensionality of the test function) )
        list: corresponding robustness
        list: samples from acquisition function for reuse in classification
    """

    samples_in_new = []
    acquisition_fun_final_samples = []
    corresponding_robustness_new = []
    for i in range(samples_in.shape[0]):
        X = samples_in[i,:,:]
        Y = corresponding_robustness[i,:].reshape((corresponding_robustness.shape[1],1))
        acquisition_fun_sample_region = []
        for j in range(number_of_samples_to_generate[i]):
            model = GaussianProcessRegressor()
            model.fit(X, Y)
            
            min_bo, samples_acquistion = opt_acquisition(X, Y, model, num_points_to_construct_gp, test_function_dimension, region_support[i,:,:], rng)
            actual = calculate_robustness(np.array(min_bo), test_function)
            acquisition_fun_sample_region.append(samples_acquistion)
            X = np.vstack((X, np.array(min_bo)))
            Y = np.vstack((Y, np.array(actual)))
        acquisition_fun_final_samples.append(acquisition_fun_sample_region)
        samples_in_new.append(np.expand_dims(X, axis = 0))
        corresponding_robustness_new.append(np.transpose(Y))
        
    return samples_in_new, corresponding_robustness_new, acquisition_fun_final_samples


# rng = np.random.default_rng(seed)
# region_support = np.array([[[-1, 1], [-1, 1]]])
# test_function_dimension = 2
# number_of_samples = 20

# x = uniformSampling(number_of_samples, region_support, test_function_dimension, rng)
# y = calculate_robustness(x)

# x_new, y_new, s = bayesian_optimization(x, y, [10], test_function_dimension, region_support, 10, rng)
# return x_new, y_new,s



# def run_par(data):
#     num_samples, BO_samples, s = data
#     rng = np.random.default_rng(s)
#     region_support = np.array([[[-1, 1], [-1, 1]]])
#     test_function_dimension = 2
#     number_of_samples = num_samples

#     x = uniformSampling(number_of_samples, region_support, test_function_dimension, rng)
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