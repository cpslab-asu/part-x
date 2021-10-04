import numpy as np
from ..utitlies.utils_partx import plotRegion
import matplotlib.pyplot as plt
import pickle
from ..models.partx_options import partx_options
from ..numerical.classification import calculate_volume
from sklearn.gaussian_process import GaussianProcessRegressor
from ..numerical.sampling import lhs_sampling
from scipy import stats
from ..numerical.calculate_robustness import calculate_robustness
from .testFunction import callCounter
import pathlib


def load_tree_from_file(tree_name):
    f = open(tree_name, "rb")
    ftree = pickle.load(f)
    return ftree

def process_result(tree):
    pass



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
            samples = lhs_sampling(options.M, node_data.region_support, options.test_function_dimension, rng)
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
        print("{} of {} done".format(iterate, len(leaves)))
    print(np.sum(np.array(falsification_volumes),axis=0))
    return np.array(falsification_volumes)

def get_true_fv(number_of_samples, options, rng, test_function):
    callCount = callCounter(test_function)
    samples = lhs_sampling(number_of_samples, options.initial_region_support, options.test_function_dimension, rng)
    y = calculate_robustness(samples, callCount)
    # print(calculate_volume(options.initial_region_support))
    # print(np.sum(np.array(y <= 0)))
    # print(number_of_samples)
    return (np.sum(np.array(y <= 0)) / (number_of_samples)) * calculate_volume(options.initial_region_support), samples, y

def con_int(x, conf_at):
    mean, std = x.mean(), x.std(ddof=1)
    conf_intveral = stats.norm.interval(conf_at, loc=mean, scale=std)
    return conf_intveral



def test_function(X):  ##CHANGE
    # return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 # Himmelblau's
    return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) - 20 # Rosenbrock
    # return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) + (100 * (X[2] - X[1] **2)**2 + ((1 - X[1])**2)) - 20
    # return (1 + (X[0] + X[1] + 1) ** 2 * (
    #             19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
    #                    30 + (2 * X[0] - 3 * X[1]) ** 2 * (
    #                        18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50


BENCHMARK_NAME = "Himmelblaus_3"

result_directory = pathlib.Path().joinpath('result_files').joinpath(BENCHMARK_NAME).joinpath(BENCHMARK_NAME + "_result_generating_files")

quantiles_at = [0.5, 0.95, 0.99]

f = open(result_directory.joinpath(BENCHMARK_NAME + "_options.pkl"), "rb")
options = pickle.load(f)
f.close()
start_seed = 5000
print(vars(options))
number_of_macro_replications = 50


# for i in range(number_of_macro_replications):
#     rng = np.random.default_rng(start_seed + i)
#     ftree = load_tree(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + ".pkl"))
#     falsification_volume_arrays = falsification_volume_using_gp(ftree, options, quantiles_at, rng)
    
#     f = open(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + "_fal_val_gp.pkl"), "wb")
#     pickle.dump(falsification_volume_arrays,f)
#     f.close()


volume_wo_gp_rep = []
volume_w_gp_rep = []

for i in range(number_of_macro_replications):
    f = open(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + "_fal_val_gp.pkl"), "rb")
    arr = pickle.load(f)
    f.close()
    volume_w_gp_rep.append(np.sum(np.array(arr),axis = 0))
    

    ftree = load_tree(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + ".pkl"))
    volume_wo_gp_rep.append(falsification_volume(ftree, options))

con_int_wo_gp = con_int(np.array(volume_wo_gp_rep), 0.95)
con_int_w_gp_50 = con_int(np.array(volume_w_gp_rep)[:,0], 0.95)
con_int_w_gp_95 = con_int(np.array(volume_w_gp_rep)[:,1], 0.95)
con_int_w_gp_99 = con_int(np.array(volume_w_gp_rep)[:,2], 0.95)


start_seed = 10000
rng = np.random.default_rng(start_seed)
true_fv, x,y = get_true_fv(10000, options, rng, test_function)

vol_w_gp = np.mean(volume_w_gp_rep, axis =0)
vol_wo_gp = np.mean(volume_wo_gp_rep)

print("{};{};{};{};{};{};{};{};{};{};{};{};{}".format(true_fv[0], vol_w_gp[0], con_int_w_gp_50[0], con_int_w_gp_50[1], 
                                            vol_w_gp[1], con_int_w_gp_95[0], con_int_w_gp_95[1],
                                            vol_w_gp[2], con_int_w_gp_99[0], con_int_w_gp_99[1],
                                            vol_wo_gp, con_int_wo_gp[0], con_int_wo_gp[1]))

# import matplotlib.pyplot as plt
# for iterate,i in enumerate(y[0]):
#     print(iterate)
#     if i <= 0:
#         plt.plot(x[0,iterate,0], x[0,iterate,1], 'r.')
#     else:
#         plt.plot(x[0,iterate,0], x[0,iterate,1], 'g.')
# plt.show()