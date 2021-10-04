import numpy as np
from ..numerical.sampling import lhs_sampling
from ..numerical.estimate_quantiles import estimate_quantiles
from ..numerical.bayesianOptimization import bayesian_optimization
from ..numerical.calculate_robustness import calculate_robustness
from ..numerical.classification import classification

class partx_node(object):
    def __init__(self, region_support, samples_in, samples_out, direction_of_branch, region_class = 'r'):
        
        self.region_support = region_support
        self.direction_of_branch = direction_of_branch
        self.region_class = region_class
        self.samples_in = samples_in
        self.samples_out = samples_out
        
    def generate_samples_points_grid(self, options, rng):
        self.grid = lhs_sampling(options.R * options.M, self.region_support, options.test_function_dimension, rng)

    def samples_management_unclassified(self, options, test_function, rng):
        number_of_samples_present = self.samples_out.shape[1]
        diff_number_of_samples_uniform = options.initialization_budget - number_of_samples_present

        if diff_number_of_samples_uniform > 0:
            # number_of_points_to_sample = diff_number_of_samples_uniform
            samples_in_uniform = lhs_sampling(diff_number_of_samples_uniform, self.region_support, options.test_function_dimension, rng)
            samples_out_uniform = calculate_robustness(samples_in_uniform, test_function)
            if self.samples_out.shape[1] == 0:
                new_samples_in = samples_in_uniform
                new_samples_out = samples_out_uniform
            else:
                new_samples_in = np.concatenate((self.samples_in,samples_in_uniform), axis = 1)
                new_samples_out = np.concatenate((self.samples_out,samples_out_uniform), axis = 1)
        else:
            new_samples_in = self.samples_in
            new_samples_out = self.samples_out
        # print(new_samples_in)
        # print(new_samples_out)


        #insert sbo points
        samples_fr_grid_for_gp = self.grid[:,0:options.NGP,:]
        final_new_samples_in, final_new_samples_out = bayesian_optimization(test_function, new_samples_in, new_samples_out, options.number_of_BO_samples, options.test_function_dimension, self.region_support, samples_fr_grid_for_gp, rng)
        self.samples_in = final_new_samples_in[0]
        self.samples_out = final_new_samples_out[0]
        # self.bo_samples = []
        return final_new_samples_in, final_new_samples_out
    
    def samples_management_classified(self, options, test_function, number_of_samples, rng):
        samples_uniform_in = lhs_sampling(number_of_samples, self.region_support, options.test_function_dimension, rng)
        samples_uniform_out = calculate_robustness(samples_uniform_in, test_function)
        self.samples_in = np.concatenate((self.samples_in, samples_uniform_in), axis=1)
        self.samples_out = np.concatenate((self.samples_out, samples_uniform_out), axis=1)
        
        return  self.samples_in,  self.samples_out

    def calculate_and_classifiy(self, options,rng):
        
        self.lower_bound, self.upper_bound = estimate_quantiles(self.samples_in, self.samples_out, self.grid, self.region_support, options.test_function_dimension, options.alpha,options.R,options.M, rng)
        
        self.region_class = classification(self.region_support, self.region_class, options.min_volume, self.lower_bound, self.upper_bound)
        return self.region_class
        