import numpy as np
from .exp_statistics import load_tree, falsification_volume, con_int
from ..models.partx_options import partx_options
import pathlib
import pickle
from ..kriging_gpr.interface.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from ..kriging_gpr.interface.OK_Rpredict import OK_Rpredict
import pandas as pd 

def cal_std_err(x):
    num_macro_rep = len(x)
    # print(num_macro_rep)
    unbiased_std_dev = (np.sum((x - np.mean(x))**2))/(num_macro_rep-1)
    std_err = unbiased_std_dev / num_macro_rep
    return std_err

def generate_statistics(BENCHMARK_NAME, number_of_macro_replications, quantiles_at, confidence_at, folder_name):
    result_directory = pathlib.Path().joinpath(folder_name).joinpath(BENCHMARK_NAME).joinpath(BENCHMARK_NAME + "_result_generating_files")


    f = open(result_directory.joinpath(BENCHMARK_NAME + "_options.pkl"), "rb")
    options = pickle.load(f)
    f.close()

    result_dictionary = {}
    
    volume_wo_gp_rep_classified = []
    volume_wo_gp_rep_unclassified = []
    volume_w_gp_rep = []
    first_falsification = []
    falsification_corresponding_points = []
    unfalsification_corresponding_points = []
    falsified_true = []
    best_robustness = []
    best_robustness_points = []
    fr_count = 0
    for i in range(number_of_macro_replications):
        f = open(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + "_fal_val_gp.pkl"), "rb")
        arr = pickle.load(f)
        f.close()
        volume_w_gp_rep.append(np.sum(np.array(arr),axis = 0))
        
        f = open(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + "_point_history.pkl"), "rb")
        point_history = pickle.load(f)
        f.close()
        # print(point_history)
        
        point_history = np.array(point_history)
        list_of_neg_rob = np.where(point_history[:,-1] <= 0)
        list_of_pos_rob = np.where((point_history[:,-1] > 0))
       
        best_robustness.append(np.min(point_history[:,-1]))
        best_robustness_points.append(point_history[np.argmin(point_history[:,-1])])
        if list_of_neg_rob[0].size > 0 :
            falsified_true.append(1)
            fr_count = fr_count + 1
            first_falsification.append(point_history[list_of_neg_rob[0][0],0])
            falsification_corresponding_points.append(point_history[list_of_neg_rob[0][0]])
        else:
            # unfalsification_corresponding_points.append(point_history[list_of_pos_rob[0][0]])
            unfal_ind = np.argmin(point_history[list_of_pos_rob,-1])
            unfalsification_corresponding_points.append(point_history[unfal_ind])
            falsified_true.append(0)
            first_falsification.append(options.max_budget)

        ftree = load_tree(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + ".pkl"))
        vol_classified, vol_unclassified = falsification_volume(ftree, options)
        volume_wo_gp_rep_classified.append(vol_classified)
        volume_wo_gp_rep_unclassified.append(vol_unclassified)

    result_generating_dictionary_for_verif = {
        "volume_wo_gp_rep_classified" : volume_wo_gp_rep_classified,
        "volume_wo_gp_rep_unclassified": volume_wo_gp_rep_unclassified,
        "volume_w_gp_rep" : volume_w_gp_rep
    }
    volume_w_gp_rep = np.array([volume_w_gp_rep[i] for i in range(len(volume_w_gp_rep))])

    f = open(result_directory.joinpath(BENCHMARK_NAME + "_arrays_for_verif_result.pkl"), "wb")
    pickle.dump(result_generating_dictionary_for_verif, f)
    f.close
    
    con_int_wo_gp_classified = con_int(np.array(volume_wo_gp_rep_classified), confidence_at)
    con_int_wo_gp_unclassified = con_int(np.array(volume_wo_gp_rep_unclassified), confidence_at)
    vol_wo_gp_classified = np.mean(volume_wo_gp_rep_classified)
    vol_wo_gp_unclassified = np.mean(volume_wo_gp_rep_unclassified)
    vol_wo_gp_classified_sd = cal_std_err(volume_wo_gp_rep_classified)
    vol_wo_gp_unclassified_sd = cal_std_err(volume_wo_gp_rep_unclassified)

    fv_wo_gp = np.array([[vol_wo_gp_classified,  vol_wo_gp_classified_sd, con_int_wo_gp_classified[0], con_int_wo_gp_classified[1]],
                [vol_wo_gp_unclassified,  vol_wo_gp_unclassified_sd, con_int_wo_gp_unclassified[0], con_int_wo_gp_unclassified[1]]])
    
    fv_wo_gp_table = pd.DataFrame(fv_wo_gp, index = ['Classified Regions only', 'Classified + Unclassified Regions'], 
                                    columns=['Mean', 'Std Error', 'UCB', 'LCB'])
    vol_w_gp = np.mean(volume_w_gp_rep, axis =0)
    vol_w_gp_sd = [cal_std_err(volume_w_gp_rep[:,i]) for i in range(len(quantiles_at))]

    fv_stats_complete = []
    for iterate in range(len(quantiles_at)):
        conf_interval = con_int(np.array(volume_w_gp_rep)[:,iterate], confidence_at)
        fv_stats_temp = [vol_w_gp[iterate], vol_w_gp_sd[iterate], conf_interval[0], conf_interval[1]]
        fv_stats_complete.append(fv_stats_temp)
    fv_stats_with_gp = np.array(fv_stats_complete)

    fv_stats_with_gp_table = pd.DataFrame(data = fv_stats_with_gp, index = quantiles_at, columns = ['Mean', 'Std Error', 'UCB', 'LCB'])
    result_dictionary['fv_stats_with_gp'] = fv_stats_with_gp_table
    result_dictionary['fv_stats_wo_gp'] = fv_wo_gp_table



    result_dictionary['falsified_true'] = falsified_true
    falsification_rate = np.sum(falsified_true)
    numpoints_fin_first_f_mean = np.mean(first_falsification)
    numpoints_fin_first_f_min = np.min(first_falsification)
    numpoints_fin_first_f_max = np.max(first_falsification)
    numpoints_fin_first_f_median = np.median(first_falsification)
    result_dictionary['first_falsification_mean'] = numpoints_fin_first_f_mean
    result_dictionary['first_falsification_median'] = numpoints_fin_first_f_median
    result_dictionary['first_falsification_min'] = numpoints_fin_first_f_min
    result_dictionary['first_falsification_max'] = numpoints_fin_first_f_max
    result_dictionary['falsification_rate'] = falsification_rate
    result_dictionary['best_robustness'] = np.min(best_robustness)

    
    result_dictionary["first_falsification_point"] = falsification_corresponding_points
    result_dictionary["non_falsification_points"] = unfalsification_corresponding_points
    result_dictionary["best_falsification_points"] = best_robustness_points
    f = open(result_directory.joinpath(BENCHMARK_NAME + "_all_result.pkl"), "wb")
    pickle.dump(result_dictionary, f)
    f.close
    
    return result_dictionary
