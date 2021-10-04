import numpy as np
from .exp_statistics import load_tree, falsification_volume, con_int
from ..models.partx_options import partx_options
import pathlib
import pickle

def generate_statistics(BENCHMARK_NAME, number_of_macro_replications, quantiles_at, confidence_at, folder_name):
    result_directory = pathlib.Path().joinpath(folder_name).joinpath(BENCHMARK_NAME).joinpath(BENCHMARK_NAME + "_result_generating_files")

    print(result_directory.joinpath(BENCHMARK_NAME + "_options.pkl"))
    f = open(result_directory.joinpath(BENCHMARK_NAME + "_options.pkl"), "rb")
    options = pickle.load(f)
    f.close()

    print(vars(options))

    # f = open(result_directory.joinpath(BENCHMARK_NAME + "_uniform_random_results.pkl"), "rb")
    # mc_uniform_test_function = pickle.load(f)
    # f.close()

    # result_dictionary = {"true_fv" : mc_uniform_test_function["true_fv"][0]}
    result_dictionary = {"true_fv" : 0}
    
    volume_wo_gp_rep_classified = []
    volume_wo_gp_rep_unclassified = []
    volume_w_gp_rep = []
    first_falsification = []
    falsified_true = []
    best_robustness = []
    for i in range(number_of_macro_replications):
        f = open(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + "_fal_val_gp.pkl"), "rb")
        arr = pickle.load(f)
        f.close()
        volume_w_gp_rep.append(np.sum(np.array(arr),axis = 0))
        
        f = open(result_directory.joinpath(BENCHMARK_NAME + "_" + str(i) + "_point_history.pkl"), "rb")
        point_history = pickle.load(f)
        f.close()
        
        point_history = np.array(point_history)
        list_of_neg_rob = np.where(point_history[:,-1] < 0)
        best_robustness.append(np.min(point_history[:,-1]))
        # print(list_of_neg_rob[0][0])
        # print(list_of_neg_rob)
        if list_of_neg_rob[0] != []:
            falsified_true.append(1)
            first_falsification.append(point_history[list_of_neg_rob[0][0],0])
        else:
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
    f = open(result_directory.joinpath(BENCHMARK_NAME + "_arrays_for_verif_result.pkl"), "wb")
    pickle.dump(result_generating_dictionary_for_verif, f)
    f.close
    
    con_int_wo_gp_classified = con_int(np.array(volume_wo_gp_rep_classified), confidence_at)
    con_int_wo_gp_unclassified = con_int(np.array(volume_wo_gp_rep_unclassified), confidence_at)
    vol_wo_gp_classified = np.mean(volume_wo_gp_rep_classified)
    vol_wo_gp_unclassified = np.mean(volume_wo_gp_rep_unclassified)
    vol_wo_gp_classified_sd = np.std(volume_wo_gp_rep_classified)
    vol_wo_gp_unclassified_sd = np.std(volume_wo_gp_rep_unclassified)

    confidence_at_string = str(confidence_at).replace(".","_")

    result_dictionary["con_int_fv_wo_gp_classified_quan_confidence_"+confidence_at_string] = con_int_wo_gp_classified
    result_dictionary["con_int_fv_wo_gp_classified_unclassified_quan_confidence_"+confidence_at_string] = con_int_wo_gp_unclassified
    result_dictionary["mean_fv_wo_gp_classified"] = vol_wo_gp_classified
    result_dictionary["mean_fv_wo_gp_classified_unclassified"] = vol_wo_gp_unclassified
    result_dictionary["std_dev_fv_wo_gp_classified"] = vol_wo_gp_classified_sd
    result_dictionary["std_dev_fv_wo_gp_classified_unclassified"] = vol_wo_gp_unclassified_sd


    con_int_w_gp_50 = con_int(np.array(volume_w_gp_rep)[:,0], confidence_at)
    con_int_w_gp_95 = con_int(np.array(volume_w_gp_rep)[:,1], confidence_at)
    con_int_w_gp_99 = con_int(np.array(volume_w_gp_rep)[:,2], confidence_at)

    vol_w_gp = np.mean(volume_w_gp_rep, axis =0)
    vol_w_gp_sd = np.std(volume_w_gp_rep, axis =0)
    
    result_dictionary['falsified_true'] = falsified_true
    falsification_rate = np.sum(falsified_true)
    numpoints_fin_first_f_mean = np.mean(first_falsification)
    numpoints_fin_first_f_min = np.min(first_falsification)
    numpoints_fin_first_f_max = np.max(first_falsification)
    numpoints_fin_first_f_median = np.median(first_falsification)
    result_dictionary['numpoints_fin_first_f_mean'] = numpoints_fin_first_f_mean
    result_dictionary['numpoints_fin_first_f_median'] = numpoints_fin_first_f_median
    result_dictionary['numpoints_fin_first_f_min'] = numpoints_fin_first_f_min
    result_dictionary['numpoints_fin_first_f_max'] = numpoints_fin_first_f_max
    result_dictionary['falsification_rate'] = falsification_rate
    result_dictionary['best_robustness'] = np.min(best_robustness)
    for iterate in range(len(quantiles_at)):
        quantile_string = str(quantiles_at[iterate]).replace(".","_")
        result_dictionary["mean_fv_with_gp_quan" + quantile_string] = vol_w_gp[iterate]
        result_dictionary["std_dev_fv_with_gp_quan" + quantile_string] = vol_w_gp_sd[iterate]
        conf_interval = con_int(np.array(volume_w_gp_rep)[:,iterate], confidence_at)
        result_dictionary["con_int_fv_with_gp_quan_"+quantile_string+"_confidence_"+confidence_at_string] = conf_interval

    f = open(result_directory.joinpath(BENCHMARK_NAME + "_all_result.pkl"), "wb")
    pickle.dump(result_dictionary, f)
    f.close
    
    
    # import matplotlib.pyplot as plt
    # for iterate,i in enumerate(y[0]):
    #     print(iterate)
    #     if i <= 0:
    #         plt.plot(x[0,iterate,0], x[0,iterate,1], 'r.', markersize = 4)
    #     else:
    #         plt.plot(x[0,iterate,0], x[0,iterate,1], 'g.', markersize = 4)
    # plt.show()

    # print("{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(true_fv[0], 
    #                                             vol_w_gp[0], con_int_w_gp_50[0], con_int_w_gp_50[1], 
    #                                             vol_w_gp[1], con_int_w_gp_95[0], con_int_w_gp_95[1],
    #                                             vol_w_gp[2], con_int_w_gp_99[0], con_int_w_gp_99[1],
    #                                             vol_wo_gp_classified, con_int_wo_gp_classified[0], con_int_wo_gp_classified[1],
    #                                             vol_wo_gp_unclassified, con_int_wo_gp_unclassified[0], con_int_wo_gp_unclassified[1]))

    # print("**********")
    # # print("{};{};{}".format(vol_wo_gp_unclassified, con_int_wo_gp_unclassified[0], con_int_wo_gp_unclassified[1]))
    

    # print("{}\n{}\n{}\n{}".format(vol_wo_gp_classified_sd, vol_w_gp_sd[0], vol_w_gp_sd[1], vol_w_gp_sd[2]))
    # print("***************")
    # print(result_dictionary)
    return result_dictionary