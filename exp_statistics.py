import numpy as np
from utils_partx import plotRegion
import matplotlib.pyplot as plt
import pickle
from partx_options import partx_options
from classification import calculate_volume
from sklearn.gaussian_process import GaussianProcessRegressor
from sampling import uniformSampling
from scipy import stats
from calculate_robustness import calculate_robustness

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

def falsification_volume_using_gp(ftree, options, quantiles_at, exp_name):
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
            for x in range(options.M):
                quantiles_values_alp = []
                for alp in quantiles_at:
                    
                    quantiles_values = (stats.norm.ppf(alp,y_pred[x][0],sigma_st[x]))
                    print(quantiles_values)
                    quantiles_values_alp.append(quantiles_values)
                quantile_values_m.append(quantiles_values_alp)
            quantile_values_r.extend(quantile_values_m)
        falsified_volume_region = ((np.array(quantile_values_r) < 0).sum(axis=0) / (options.R*options.M)) * calculate_volume(node_data.region_support)
        falsification_volumes.append(falsified_volume_region)
        print("{} of {} done".format(iterate, len(leaves)))
    print(np.sum(np.array(falsification_volumes),axis=0))
    return np.array(falsification_volumes)

def get_true_fv(number_of_samples, options):
    samples = uniformSampling(number_of_samples, options.initial_region_support, options.test_function_dimension)
    y = calculate_robustness(samples)
    print(calculate_volume(options.initial_region_support))
    print(np.sum(np.array(y <= 0)))
    print(number_of_samples)
    return (np.sum(np.array(y <= 0)) / (number_of_samples)) * calculate_volume(options.initial_region_support)

function_name = "Goldstein_Price"
exp_name = function_name + "_3"
f = open(exp_name+"_options.pkl", "rb")
options = pickle.load(f)
f.close()

ftree = load_tree(exp_name+".pkl")
print(vars(options))
# leaves = ftree.leaves()
# for x,i in enumerate(leaves):
#     # fig = plt.figure()
#     x_1, y_1, x_2,y_2,x_3,y_3,x_4,y_4 = plotRegion(i.data.region_support)
#     plt.plot(x_1,y_1)
#     plt.plot(x_2,y_2)
#     plt.plot(x_3,y_3)
#     plt.plot(x_4,y_4)
    # leaves = ftree.leaves()
# for x,i in enumerate(leaves):
#     # fig = plt.figure()
#     x_1, y_1, x_2,y_2,x_3,y_3,x_4,y_4 = plotRegion(i.data.region_support)
#     plt.plot(x_1,y_1)
#     plt.plot(x_2,y_2)
#     plt.plot(x_3,y_3)
#     plt.plot(x_4,y_4)
    
#     if i.data.region_class == "+":
#         plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'g.')
#     elif i.data.region_class == "-":
#         plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'r.')
# # plt.title("{} Function Budget = {} -- BO Grid {} x {}".format(function_name, options.max_budget, number_of_BO_samples[0], number_of_samples_gen_GP))
# plt.show()
#     if i.data.region_class == "+":
#         plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'g.')
#     elif i.data.region_class == "-":
#         plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'r.')
# # plt.title("{} Function Budget = {} -- BO Grid {} x {}".format(function_name, options.max_budget, number_of_BO_samples[0], number_of_samples_gen_GP))
# plt.show()
# volume_gp = falsification_volume_using_gp(ftree, options, [0.5,0.9, 0.95, 0.99], exp_name)
# print("****************")
volume = falsification_volume(ftree, options)
f = open(exp_name+"_falsification_volume_gp.pkl", "rb")
volume_gp = pickle.load(f)
f.close()
print("****************")
x = (np.sum(np.array(volume_gp),axis = 0))
true_fv = get_true_fv(10000, options)
print("{}\t{}\t{}\t{}\t{}\t{}".format(true_fv[0], x[0],x[1],x[2],x[3],volume))
# import matplotlib.pyplot as plt
# for iterate,i in enumerate(y[0]):
#     print(iterate)
#     if i <= 0:
#         plt.plot(x[0,iterate,0], x[0,iterate,1], 'r.')
#     else:
#         plt.plot(x[0,iterate,0], x[0,iterate,1], 'g.')
# plt.show()