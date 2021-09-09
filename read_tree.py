import matplotlib.pyplot as plt
import pickle

function_name = "Goldstein_Price"
exp_name = function_name + "_1"
f = open(exp_name + ".pkl", "rb")
ftree = pickle.load(f)

from utils_partx import plotRegion
ftree.show()



leaves = ftree.leaves()
print("number of leaves= {}".format(len(leaves)))
print("******************")
# plt.ion()

print("*******************************************************")
for x,i in enumerate(leaves):
    # fig = plt.figure()
    x_1, y_1, x_2,y_2,x_3,y_3,x_4,y_4 = plotRegion(i.data.region_support)
    plt.plot(x_1,y_1)
    plt.plot(x_2,y_2)
    plt.plot(x_3,y_3)
    plt.plot(x_4,y_4)

    
    if i.data.region_class == "+":
        plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'g.')
    elif i.data.region_class == "-":
        plt.plot(i.data.samples_in[0,:,0], i.data.samples_in[0,:,1], 'r.')

plt.show()