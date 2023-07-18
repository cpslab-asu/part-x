from typing import Callable
import numpy as np
from numpy.typing import NDArray

from ..sampling import lhs_sampling, uniform_sampling, OOBError
from ..quantileClassification import estimate_quantiles, classification
from ..utils import compute_robustness
from ..bayesianOptimization import BOSampling


class PartXNode:
    def __init__(self, self_id: int, parent_id: int, region_support: NDArray, samples_in: NDArray, samples_out: NDArray, branch_dir:int, region_class: str = 'r'):
        """PartXNode Operations for various types of region classifications.

        Args:
            self_id: Identifier of Node to allow identification of bugs after tree is created.
            parent_id: Identifier of Parent Node to allow identification of bugs after tree is created.
            region_support: Min and Max of all dimensions
            samples_in: Samples from Training set.
            samples_out: Evaluated values of samples from Training set.
            branch_dir: Direction in which this node should be brached (if needed).
            region_class (str, optional): Current Type of region. Defaults to 'r'.
        """

        self.self_id = self_id
        self.parent_id = parent_id
        self.region_support = region_support
        self.region_class = region_class
        self.samples_in = samples_in
        self.samples_out = samples_out
        self.branch_dir = branch_dir


        # if self.parent_id != 0 and self.samples_in.shape[0] == 0:
        #     self.region_class = "u"
        

    def samples_management_unclassified(self, test_function: Callable, options, oracle_info, rng):
        """Method to manage samples in subregion which is unclassified (r, r+, r-)

        Args:
            test_function: Function of System Under Test.
            options: PartX Options object
            rng: RNG object from numpy

        Raises:
            ValueError: If options.init_sampling_type is not defined correctly.

        Returns:
            Class of new region
        """
        
        assert self.region_class == "r" or self.region_class == "r+" or self.region_class == "r-"
        samples_present = self.samples_out.shape[0]
        init_sampling_left = options.init_budget - samples_present
        
        try:
            if init_sampling_left > 0:
                if options.init_sampling_type == "lhs_sampling":
                    x_init_extra = lhs_sampling(init_sampling_left, self.region_support, options.tf_dim, oracle_info, rng)
                elif options.init_sampling_type == "uniform_sampling":
                    x_init_extra = uniform_sampling(init_sampling_left, self.region_support, options.tf_dim, oracle_info, rng)
                else:
                    raise ValueError(f"{options.init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
                
                y_init_extra = compute_robustness(x_init_extra, test_function)
                if self.samples_out.shape[0] == 0:
                    new_samples_in = x_init_extra
                    new_samples_out = y_init_extra
                else:
                    new_samples_in = np.concatenate((self.samples_in,x_init_extra), axis = 0)
                    new_samples_out = np.concatenate((self.samples_out,y_init_extra), axis = 0)
            else:
                new_samples_in = self.samples_in
                new_samples_out = self.samples_out

            bo = BOSampling(options.bo_model)
            final_new_samples_in, final_new_samples_out, _, _ = bo.sample(test_function, options.bo_budget, new_samples_in, new_samples_out, self.region_support, options.gpr_model, oracle_info, rng)
            self.samples_in = final_new_samples_in
            self.samples_out = final_new_samples_out

            self.lower_bound, self.upper_bound = estimate_quantiles(self.samples_in, self.samples_out, self.region_support, options.tf_dim, options.alpha,options.R,options.M, options.gpr_model, oracle_info, rng, options.q_estim_sampling)
            
            self.new_region_class = classification(self.region_support, self.region_class, options.min_volume, self.lower_bound, self.upper_bound)
            self.region_class = self.new_region_class
        except OOBError:
            self.new_region_class = "i"
            self.region_class = "i"
        return self.new_region_class
    
    def samples_management_classified(self, num_samples: int, test_function: Callable, options, oracle_info, rng, fin_cs = False):
        """Method to manage samples in where continued sampling is to be performed.

        Args:
            num_samples: Number of samples
            test_function: Function of System Under Test.
            options: PartX Options object
            rng: RNG object from numpy
            fin_cs: If False, performing continued sampling requires regions class be classified (+, -),  Defaults to False.

        Raises:
            ValueError: options.cs_sampling_type not correctly defined

        Returns:
            Class of new region
        """
        if not fin_cs:
            assert self.region_class == "+" or self.region_class == "-"
        else:
            assert self.region_class == "r" or self.region_class == "r+" or self.region_class == "r-" or self.region_class == "+" or self.region_class == "-"
        
        if options.cs_sampling_type == "lhs_sampling":
            cs_samples_in = lhs_sampling(num_samples, self.region_support, options.tf_dim, oracle_info, rng)
        elif options.cs_sampling_type == "uniform_sampling":
            cs_samples_in = uniform_sampling(num_samples, self.region_support, options.tf_dim, oracle_info, rng)
        else:
            raise ValueError(f"{options.cs_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")


        cs_samples_out = compute_robustness(cs_samples_in, test_function)
        self.samples_in = np.concatenate((self.samples_in, cs_samples_in), axis=0)
        self.samples_out = np.concatenate((self.samples_out, cs_samples_out), axis=0)
        
        self.lower_bound, self.upper_bound = estimate_quantiles(self.samples_in, self.samples_out, self.region_support, options.tf_dim, options.alpha,options.R,options.M, options.gpr_model, oracle_info, rng)
        
        self.new_region_class = classification(self.region_support, self.region_class, options.min_volume, self.lower_bound, self.upper_bound)
        self.region_class = self.new_region_class

        return self.new_region_class

    
        
        
