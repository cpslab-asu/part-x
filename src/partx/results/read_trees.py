import numpy as np
import treelib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# from ..utils import calculate_volume
def save_trees_plots(ftree, options):
    
    

    leaves = ftree.leaves()
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    # print("*******************************************************")
    points_in_list = []
    node_id = []
    points_class = []
    print(options.init_reg_sup)
    ax.set_xlim(options.init_reg_sup[0,0], options.init_reg_sup[0,1])
    ax.set_ylim(options.init_reg_sup[1,0], options.init_reg_sup[1,1]) 
    for x,i in enumerate(leaves):
        # print(f"Reg_class = {i.data.region_class} Volume = {calculate_volume(i.data.region_support)}, min_vol = {options.min_volume}")
        r_support = np.array(i.data.region_support)
        x = r_support[0,0]
        y = r_support[1,0]
        w = r_support[0,1] - r_support[0,0]
        h = r_support[1,1] - r_support[1,0]

        if y+h != options.init_reg_sup[0,1]:
            if x+w != options.init_reg_sup[1,1] or x != options.init_reg_sup[1,0]:
                val_x = [r_support[0,0],  r_support[0,1]]
                val_y = [r_support[1,1], r_support[1,1]]
                plt.plot(val_x, val_y, 'k', linewidth = 1.5)
        
        if x+w != options.init_reg_sup[1,1]:
            if y+h != options.init_reg_sup[0,1] or y != options.init_reg_sup[0,0]:
                val_y = [r_support[1,0],  r_support[1,1]]
                val_x = [r_support[0,1], r_support[0,1]]
                plt.plot(val_x, val_y, 'k', linewidth = 1.5)
        

        points_class.append(i.data.region_class)
        points_in_list.append((i.data.samples_in).shape[1])
        node_id.append(i.identifier)
        print(i.data.samples_in[:,0])
        if i.data.region_class == "+":
            ax.add_patch( Rectangle((x,y),
                            w, h,
                            fc ='green',
                            alpha = 0.4))
            ax.plot(i.data.samples_in[:,0], i.data.samples_in[:,1], 'g.', markersize = 2)
        elif i.data.region_class == "-":
            ax.add_patch( Rectangle((x,y),
                            w, h,
                            ec = 'black',
                            fc ='red',
                            alpha = 0.4))
            ax.plot(i.data.samples_in[:,0], i.data.samples_in[:,1], 'r.', markersize = 2)
        elif i.data.region_class == "r" or i.data.region_class == "r+" or i.data.region_class == "r-":
            ax.add_patch( Rectangle((x,y),
                            w, h,
                            ec = 'black',
                            fc ='blue',
                            alpha = 0.4))
            ax.plot(i.data.samples_in[:,0], i.data.samples_in[:,1], 'b.', markersize = 2)
    plt.title("Constraints such that anything in circle at (-1,0) with radius 0.18 is invalid")
    plt.savefig("Goldstein_cs_5.png", dpi = 400, bbox_inches = 'tight')
    
import pickle

with open("NLF_GoldsteinPrice_Constraints/Goldstein_final/Goldstein_final_result_generating_files/Goldstein_final_0.pkl", "rb") as f:
    t = pickle.load(f)

with open("NLF_GoldsteinPrice_Constraints/Goldstein_final/Goldstein_final_result_generating_files/Goldstein_final_options.pkl", "rb") as f:
    o = pickle.load(f)

save_trees_plots(t,o)