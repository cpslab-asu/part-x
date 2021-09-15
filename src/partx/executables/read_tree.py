import matplotlib.pyplot as plt
import pickle
import numpy as np
from ..models.partx_options import partx_options

def save_trees_plots(q, exp_name, options):
    
    f = open(exp_name +  ".pkl", "rb")
    ftree = pickle.load(f)
    f.close()

    leaves = ftree.leaves()
    fig = plt.figure()

    # print("*******************************************************")
    points_in_list = []
    node_id = []
    points_class = []
    for x,i in enumerate(leaves):
        # fig = plt.figure()
        x_1, y_1, x_2,y_2,x_3,y_3,x_4,y_4 = plotRegion(i.data.region_support)
        plt.plot(x_1,y_1)
        plt.plot(x_2,y_2)
        plt.plot(x_3,y_3)
        plt.plot(x_4,y_4)
        points_class.append(i.data.region_class)
        points_in_list.append((i.data.samples_in).shape[1])
        node_id.append(i.identifier)
        if i.data.region_class == "+":
            plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'g.')
        elif i.data.region_class == "-":
            plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'r.')

    plt.title("{} Function Budget = {} -- BO Grid {} x {}".format(function_name, options.max_budget, options.number_of_BO_samples[0], options.number_of_samples_gen_GP))
    plt.savefig(exp_name+".png")
    print("*****************************")
    print("Points in Replication {} = {}".format(q, sum(points_in_list)))

# dir_name = "Rosenbrock_1/"
# function_name = "Rosenbrock"

# dir_name = "Goldstein_price_1/"
# function_name = "Goldstein_price"

dir_name = "Himmelblaus_2/"
function_name = "Himmelblaus"

f = open(dir_name+ function_name + "_2_options.pkl", "rb")
options = pickle.load(f)
f.close()
from utils_partx import plotRegion

for i in range(50):
    exp_name = function_name + "_2_" + str(i)
    save_trees_plots(i, dir_name + exp_name, options)
