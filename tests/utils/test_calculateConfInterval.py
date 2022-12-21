import pickle
import numpy as np
import unittest
from partx.utils import conf_interval
from partx.sampling import uniform_sampling

class TestCalculateConfInterval(unittest.TestCase):
    def test1_calculate_conf_interval(self):
        with open("./tests/utils/goldResources/Testing_123_arrays_for_verif_result.pkl", "rb") as f:
            gr_array = pickle.load(f)

        gr = volume_w_gp_rep = gr_array["volume_w_gp_rep"]
        quantiles_at = [0.5, 0.05, 0.01]
        confidence_at = 0.95
        con_ints = []
        for iterate in range(len(quantiles_at)):
            con_int = conf_interval(np.array(volume_w_gp_rep)[:,iterate], confidence_at)
            con_ints.append(con_int)

        con_ints = np.array(con_ints)

        with open("./tests/utils/goldResources/Testing_123_arrays_for_verif_result_con_int.pkl", "rb") as f:
            # pickle.dump(con_ints, f)
            gr_con_int = pickle.load(f) 

        np.testing.assert_array_equal(con_ints, gr_con_int)

        

        