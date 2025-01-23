import numpy as np
import numpy.typing as npt
import time
import pickle
from dataclasses import dataclass
import matplotlib.pyplot as plt


def branch_region(region_support: np.array, direction_of_branching:int, uniform: bool,branching_factor:int, rng) -> np.array:
    """Generate new region supports based on direction of branching and the branching factor. For now, the
    partitioning of space is uniformly done.

    Args:
        region_support: The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        direction_of_branching: Dimension or direction in which branching is to be done
        uniform: Do we partition the region uniformly or randomly. If True, uniform partition takes place
        branching_factor: How many new region supports to be made from the previous region support

    Returns:
        np.array: New region support
    """

    dim_length = region_support[direction_of_branching][1] - region_support[direction_of_branching][0]
    if uniform:
        split_array = region_support[direction_of_branching][0] + (np.arange(branching_factor+1)/branching_factor) * dim_length
    else: 
        split_array = region_support[direction_of_branching][0] + np.sort(np.insert(rng.uniform(0,1,branching_factor-1), 0,[0,1])) * dim_length
        # print(split_array)
    
    
    new_bounds = []
    for i in range(branching_factor):
        temp = region_support.copy()
        temp[direction_of_branching,0] = split_array[i]
        temp[direction_of_branching,1] = split_array[i+1]
        
        new_bounds.append(temp)
    return new_bounds



def calculate_volume(region_support: npt.NDArray) -> list:
    """Calculate volume of a hypercube. 

    Args:
        region_support: The bounds of the region within which the sampling is to be done.
                                    Region Bounds is N x O where;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;

    Returns:
        float: volume. List of length = number of regions
    """
    return np.prod(region_support[:,1]-region_support[:,0], axis = 0)


def compute_robustness(samples_in: npt.NDArray, test_function: Type[Fn]) -> npt.NDArray:
    """Compute the fitness (robustness) of the given sample.

    Args:
        samples_in: Samples points for which the fitness is to be computed.
        test_function: Test Function insitialized with Fn
    Returns:
        Fitness (robustness) of the given sample(s)
    """

    if samples_in.shape[0] == 1:
        samples_out = np.array([test_function(samples_in[0])])
    else:
        samples_out = np.apply_along_axis(
            lambda sample: test_function(sample), 1, samples_in
        )
    return samples_out


class Fn:
    def __init__(self, func):
        self.func = func
        self.count = 0
        self.point_history = []
        self.simultation_time = []

    def __call__(self, *args, **kwargs):
        self.count = self.count + 1
        sim_time_start = time.perf_counter()
        rob_val = self.func(*args, **kwargs)
        time_elapsed = time.perf_counter() - sim_time_start
        self.simultation_time.append(time_elapsed)
        self.point_history.append([self.count, *args, rob_val])
        return rob_val


def load_tree(tree_name):
    """Load the tree

    Args:
        tree_name ([type]): Load a tree for a particular replication

    Returns:
        [type]: tree
    """
    with open(tree_name, "rb") as f:
        ftree = pickle.load(f)
    # f.close()
    return 



@dataclass
class OracleResult:
    val: float
    sat: bool

class OracleCreator:
    def __init__(self, oracle_function, n_tries_randomsampling, n_tries_BO):

        """Helps to set up options for Part-X

        Args:
           
        """
        self.oracle_function = oracle_function
        self.n_tries_randomsampling = n_tries_randomsampling
        self.n_tries_BO = n_tries_BO

        
    def __call__(self, X):
        if self.oracle_function is not None:
            val = self.oracle_function(X)
            sat = val <= 0.0
        else:
            val = -np.inf
            sat = True
        return OracleResult(val, sat)
        

def divide_points(samples_in: np.array, samples_out:np.array, region_support: list) -> list:
    """

    Args:
        samples_in: Samples from Training set.
        samples_out: Evaluated values of samples from Training set.
        region_support: Min and Max of all dimensions

    Returns:
        list: Divided samples
    """    
    regionSamples = []
    corresponding_robustenss = []
    if samples_in.shape[0] == samples_out.shape[0] and  samples_out.shape[0] != 0:

        for iterate, subregion in enumerate(region_support):
            boolArray = []
            for dimension in range(len(subregion)):
                subArray = samples_in[:, dimension]
                logical_subArray = np.logical_and(subArray >= subregion[dimension, 0], subArray <= subregion[dimension, 1])
                boolArray.append(np.squeeze(logical_subArray))
            corresponding_robustenss.append(samples_out[(np.all(boolArray, axis = 0))])
            regionSamples.append(samples_in[(np.all(boolArray, axis = 0)),:])
    else:
        for iterate, subregion in enumerate(region_support):
            corresponding_robustenss.append(np.array([]))
            regionSamples.append(np.array([[]]))
            
    return regionSamples, corresponding_robustenss
