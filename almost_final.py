from classification import calculate_volume
from utils_partx import assign_budgets, branch_new_region_support, pointsInSubRegion

from partx_node import partx_node
from partx_options import partx_options
import numpy as np
from classification import calculate_volume
import matplotlib.pyplot as plt
from budget_check import budget_check
from treelib import Tree
from calIntegral import calculate_mc_integral
from utils_partx import plotRegion
from single_replication import run_single_replication
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import logging
# from testFunction import test_function
""""
# add nugget effect to MC_integral_function
# nugget is absolute of normal distributio with mean 0 and var 0.001

# Ensemble part
"""




def test_function(X):  ##CHANGE
    # return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 # Himmelblau's
    # return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) - 20 # Rosenbrock
    return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

function_name = "Goldstein_price_3/Goldstein_price"
# function_name = "Rosenbrock_3/Rosenbrock"
# function_name = "Himmelblaus_3/Himmelblaus"
exp_name = function_name + "_3"


# Options initialization
test_function_dimension = 2
delta = 0.001
alpha = [0.95]
region_support = np.array([[[-1., 1.], [-1., 1.]]])

initialization_budget = 10
max_budget = 5000
number_of_BO_samples = [20]
number_of_samples_gen_GP = 500
continued_sampling_budget = 100
branching_factor = 2
nugget_mean = 0
nugget_std_dev = 0.001

R = number_of_BO_samples[0]
M = number_of_samples_gen_GP

options = partx_options(region_support, branching_factor, test_function_dimension, 
                        number_of_BO_samples, number_of_samples_gen_GP, alpha, M, R, 
                        delta, True, initialization_budget, max_budget, 
                        continued_sampling_budget, nugget_mean, nugget_std_dev)

f = open(exp_name + "_options.pkl", "wb")
pickle.dump(options,f)
f.close()


start_seed = 1000
inputs = []
print("Hello")
for q in range(50):
    # print(q)
    seed = start_seed + q
    # print(seed)
    # test_function.callCount = 0
    data = [q, options, exp_name+"_"+str(q), seed, test_function]
    inputs.append(data)
    # run_single_replication(q, options, exp_name+"_"+str(q), seed, test_function)
print(len(inputs))
pool = Pool()
results = list(pool.map(run_single_replication, inputs))


# leaves = ftree.leaves()


# # print("*******************************************************")
# points_in_list = []
# node_id = []
# points_class = []
# for x,i in enumerate(leaves):
#     # fig = plt.figure()
#     x_1, y_1, x_2,y_2,x_3,y_3,x_4,y_4 = plotRegion(i.data.region_support)
#     plt.plot(x_1,y_1)
#     plt.plot(x_2,y_2)
#     plt.plot(x_3,y_3)
#     plt.plot(x_4,y_4)
#     points_class.append(i.data.region_class)
#     points_in_list.append((i.data.samples_in).shape[1])
#     node_id.append(i.identifier)
#     if i.data.region_class == "+":
#         plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'g.')
#     elif i.data.region_class == "-":
#         plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'r.')
# plt.title("{} Function Budget = {} -- BO Grid {} x {}".format(function_name, options.max_budget, number_of_BO_samples[0], number_of_samples_gen_GP))
# plt.savefig(exp_name+".png")

# # print("final budget check = {}".format(test_function.callCount))

# # print(points_in_list)
# # print("*****************************")
# # print(node_id)
# # print("*****************************")
# # print(points_class)
# # print("*****************************")
# # print(sum(points_in_list))

