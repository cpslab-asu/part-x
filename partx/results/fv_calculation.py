import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats

from ..gprInterface import GPR
from ..sampling import lhs_sampling, uniform_sampling
from ..utils import calculate_volume

def fv_without_gp(ftree, options):
    """Calculate Falsification Volume Using the classified and unclassified regions

    Args:
        ftree ([type]): ftree
        options ([type]): initialization options

    Returns:
        [type]: volumes of classified and unclassified regions
    """
    leaves = ftree.leaves()
    region_supports_classified = []
    region_supports_unclassified = []
    for x,i in enumerate(leaves):
        node_data = i.data
        if node_data.region_class == "-":
            region_supports_classified.append(node_data.region_support)
        if node_data.region_class == "r" or node_data.region_class == "r+" or node_data.region_class == "r-" or node_data.region_class == "-":
            region_supports_unclassified.append(node_data.region_support)

    volumes_classified = [calculate_volume(region_supports) for region_supports in region_supports_classified]
    volumes_unclassified = [calculate_volume(region_supports) for region_supports in region_supports_unclassified]
    
    return np.sum(volumes_classified) / calculate_volume(options.init_reg_sup), np.sum(volumes_unclassified) / calculate_volume(options.init_reg_sup)

def fv_using_gp(ftree, options, oracle_info, quantiles_at, rng):
    """Calculate falsification volume using GP 

    Args:
        ftree ([type]): [description]
        options ([type]): [description]
        quantiles_at ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    leaves = ftree.leaves()
    falsification_volumes = []
    sampling_type = options.results_sampling_type
    for iterate,temp_node_id in enumerate(leaves):
        
        quantiles_falsification = np.empty((options.R, len(quantiles_at)))
        node_data = temp_node_id.data
        # 
        X = node_data.samples_in
        Y = node_data.samples_out
        if node_data.region_class != "u" and  node_data.region_class != "i":
            model = GPR(options.gpr_model)
            model.fit(X, Y)
            
            
            for r in range(options.R):
                if sampling_type == "lhs_sampling":
                    samples = lhs_sampling(options.M, node_data.region_support, options.tf_dim, oracle_info, rng)
                elif sampling_type == "uniform_sampling":
                    samples = uniform_sampling(options.M, node_data.region_support, options.tf_dim, oracle_info, rng)
                else:
                    raise ValueError(f"{sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
            
                y_pred, pred_sigma = model.predict(samples)
                for q_at_iterate in range(len(quantiles_at)):
                    q_at = quantiles_at[q_at_iterate]
                    quantiles_values = (stats.norm.ppf(q_at,y_pred,pred_sigma))
                    quantiles_falsification[r, q_at_iterate] = (np.array(quantiles_values) < 0.0).sum(axis = 0)
                
            falsified_volume_region = (quantiles_falsification.sum(axis=0) / (options.R*options.M)) * calculate_volume(node_data.region_support)
            falsification_volumes.append(falsified_volume_region)
    
    return np.array(falsification_volumes) / calculate_volume(options.init_reg_sup)
