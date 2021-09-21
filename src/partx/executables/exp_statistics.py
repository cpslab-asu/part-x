import numpy as np
# from ..utitlies.utils_partx import plotRegion
import matplotlib.pyplot as plt
import pickle
from ..models.partx_options import partx_options
from ..numerical.classification import calculate_volume
from sklearn.gaussian_process import GaussianProcessRegressor
from ..numerical.sampling import uniformSampling
from scipy import stats
from ..numerical.calculate_robustness import calculate_robustness
from ..models.testFunction import callCounter
import pathlib

def load_tree(tree_name):
    f = open(tree_name, "rb")
    ftree = pickle.load(f)
    return ftree

def falsification_volume(ftree, options):
    leaves = ftree.leaves()
    region_supports_classified = []
    region_supports_unclassified = []
    for x,i in enumerate(leaves):
        node_data = i.data
        if node_data.region_class == "-":
            region_supports_classified.append(node_data.region_support)
        if node_data.region_class == "r" or node_data.region_class == "r+" or node_data.region_class == "r-" or node_data.region_class == "-":
            region_supports_unclassified.append(node_data.region_support)

    falsified_volume_count_classified = len(region_supports_classified)
    region_supports_classified = np.reshape(np.array(region_supports_classified), (falsified_volume_count_classified,options.test_function_dimension, 2))
    volumes_classified = calculate_volume(region_supports_classified)

    falsified_volume_count_unclassified = len(region_supports_unclassified)
    region_supports_unclassified = np.reshape(np.array(region_supports_unclassified), (falsified_volume_count_unclassified,options.test_function_dimension, 2))
    volumes_unclassified = calculate_volume(region_supports_unclassified)
    return np.sum(volumes_classified), np.sum(volumes_unclassified)

def falsification_volume_using_gp(ftree, options, quantiles_at, rng):
    leaves = ftree.leaves()
    region_supports = []
    falsification_volumes = []
    for iterate,temp_node_id in enumerate(leaves):
        
        node_data = temp_node_id.data
        X = node_data.samples_in[0]
        Y = np.transpose(node_data.samples_out)
        model = GaussianProcessRegressor()
        model.fit(X, Y)
        quantile_values_r= []
        for r in range(options.R):
            samples = uniformSampling(options.M, node_data.region_support, options.test_function_dimension, rng)
            y_pred, sigma_st = model.predict(samples[0], return_std=True)
            quantile_values_m = []
            for x in range(options.M):
                quantiles_values_alp = []
                for alp in quantiles_at:
                    
                    quantiles_values = (stats.norm.ppf(alp,y_pred[x][0],sigma_st[x]))
                    # print(quantiles_values)
                    quantiles_values_alp.append(quantiles_values)
                quantile_values_m.append(quantiles_values_alp)
            quantile_values_r.extend(quantile_values_m)
        falsified_volume_region = ((np.array(quantile_values_r) < 0).sum(axis=0) / (options.R*options.M)) * calculate_volume(node_data.region_support)
        falsification_volumes.append(falsified_volume_region)
        # print("{} of {} done".format(iterate, len(leaves)))
    # print(np.sum(np.array(falsification_volumes),axis=0))
    return np.array(falsification_volumes)

def get_true_fv(number_of_samples, options, rng, test_function):
    callCount = callCounter(test_function)
    samples = uniformSampling(number_of_samples, options.initial_region_support, options.test_function_dimension, rng)
    y = calculate_robustness(samples, callCount)
    # print(calculate_volume(options.initial_region_support))
    # print(np.sum(np.array(y <= 0)))
    # print(number_of_samples)
    return (np.sum(np.array(y <= 0)) / (number_of_samples)) * calculate_volume(options.initial_region_support), samples, y

def con_int(x, conf_at):
    mean, std = x.mean(), x.std(ddof=1)
    conf_intveral = stats.norm.interval(conf_at, loc=mean, scale=std)
    return conf_intveral