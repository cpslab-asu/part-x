import numpy as np

from sampling import uniformSampling
from estimate_quantiles import estimate_quantiles
from bayesianOptimization import bayesian_optimization
from calculate_robustness import calculate_robustness
from classification import classification

region_support = np.array([[[-1, 1], [-1, 1]]])
test_function_dimension = 2
numberOfSamples = 20
R=10
M=10
delta = 0.001
alpha = [0.95]
direction_of_branch = 0
region_class = ['-','r-']

samples_in = uniformSampling(numberOfSamples, region_support, test_function_dimension)
samples_out = calculate_robustness(samples_in)
samples_in, samples_out = bayesian_optimization(samples_in, samples_out, [10], test_function_dimension, region_support, 10)

lower_bound, upper_bound = estimate_quantiles(samples_in[0], samples_out[0], region_support, 100, test_function_dimension, alpha,R,M)

print(classification(region_support, region_class, delta, lower_bound, upper_bound, direction_of_branch))