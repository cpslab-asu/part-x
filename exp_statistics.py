import numpy as np
from utils_partx import plotRegion
import matplotlib.pyplot as plt
import pickle
from partx_options import partx_options
from classification import calculate_volume
from sklearn.gaussian_process import GaussianProcessRegressor
from sampling import uniformSampling
from scipy import stats

def load_tree(tree_name):
    f = open(tree_name, "rb")
    ftree = pickle.load(f)
    return ftree

def falsification_volume(ftree, options):
    leaves = ftree.leaves()
    region_supports = []
    points_in_list = []
    node_id = []
    points_class = []
    for x,i in enumerate(leaves):
        points_class.append(i.data.region_class)
        points_in_list.append((i.data.samples_in).shape[1])
        node_id.append(i.identifier)
        node_data = i.data
        if node_data.region_class == "-":
            region_supports.append(node_data.region_support)
    falsified_volume_count = len(region_supports)
    region_supports = np.reshape(np.array(region_supports), (falsified_volume_count,options.test_function_dimension, 2))
    volumes = calculate_volume(region_supports)
    return np.sum(volumes)


def falsification_volume_using_gp(ftree, options, quantiles_at):
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
            samples = uniformSampling(options.M, node_data.region_support, options.test_function_dimension)
            y_pred, sigma_st = model.predict(samples[0], return_std=True)
            quantile_values_m = []
            for x in range(M):
                
                quantiles_values = [(stats.norm.ppf(quantile,y_pred[x],sigma_st[x]))[0] for quantile in quantiles_at]
                quantile_values_m.append(quantiles_values)
            quantile_values_r.extend(quantile_values_m)
        # print(quantile_values_r)
        falsified_volume_region = ((np.array(quantile_values_r) < 0).sum(axis=0) / (options.R*options.M)) * calculate_volume(node_data.region_support)
        # print("******************************")
        # print(falsified_regions)
        # print("****************************************************************")
        falsification_volumes.append(falsified_volume_region)
        print("{} of {} done".format(iterate, len(leaves)))
    return np.array(falsification_volumes)


# Options initialization
test_function_dimension = 2
delta = 0.001
alpha = [0.95]
region_support = np.array([[[-1., 1.], [-1., 1.]]])

initialization_budget = 10
max_budget = 5000
number_of_BO_samples = [10]
number_of_samples_gen_GP = 100
continued_sampling_budget = 100
branching_factor = 2
nugget_mean = 0
nugget_std_dev = 0.001

R = number_of_BO_samples[0]
M = number_of_samples_gen_GP


function_name = "Goldstein_Price"
exp_name = function_name + "_1"

options = partx_options(region_support, branching_factor, test_function_dimension, 
                        number_of_BO_samples, number_of_samples_gen_GP, alpha, M, R, 
                        delta, True, initialization_budget, max_budget, 
                        continued_sampling_budget, nugget_mean, nugget_std_dev)


ftree = load_tree(exp_name+".pkl")
v = falsification_volume_using_gp(ftree, options, [0.5,0.9, 0.95, 0.99])

