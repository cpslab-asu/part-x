import pickle
import numpy as np
from numpy.random import default_rng
import pickle
import unittest


from partx.gprInterface import InternalGPR
from partx.utils import Fn, compute_robustness
from partx.sampling import uniform_sampling
from partx.gprInterface import internalGPR
from partx.bayesianOptimization import BOSampling, InternalBO
from partx.utils import OracleCreator
from matplotlib import pyplot as plt

oracle_func = None

class Test_bointerface(unittest.TestCase):
    def test1_boInterface(self):
        bo = BOSampling(InternalBO())
        
        def internal_function(X):
            return X[0] ** 2 + X[1]**2 + X[2]**2

        oracle_info = OracleCreator(oracle_func, 1,1)
        rng = default_rng(12345)

        region_support = np.array([[-1, 1], [-1, 1], [-1, 1]])

        func1 = Fn(internal_function)
        in_samples_1 = uniform_sampling(20, region_support, 3, oracle_info, rng)
        out_samples_1 = compute_robustness(in_samples_1, func1)

        in_samples_2 = uniform_sampling(30, region_support, 3, oracle_info, rng)
        out_samples_2 = compute_robustness(in_samples_2, func1)

        
        
        gpr_model = InternalGPR()

        self.assertRaises(TypeError, bo.sample, func1, 50, np.array([in_samples_1]), out_samples_1, region_support, gpr_model, oracle_info,  rng)
        self.assertRaises(TypeError, bo.sample, func1, 50, in_samples_1, np.array([out_samples_1]).T, region_support, gpr_model, oracle_info, rng)
        self.assertRaises(TypeError, bo.sample, func1, 50, in_samples_1, out_samples_2, region_support, gpr_model, oracle_info, rng)
        x_complete, y_complete, x_new, y_new = bo.sample(func1, 50, in_samples_1, out_samples_1, region_support, gpr_model, oracle_info, rng)


        with open("tests/bayesianOptimization/goldResources/test_1_bo.pickle", "rb") as f:
            # pickle.dump([x_complete, y_complete, x_new, y_new], f)
            gr_x_complete, gr_y_complete, gr_x_new, gr_y_new = pickle.load(f)
        

        np.testing.assert_array_almost_equal(x_complete, gr_x_complete, decimal = 2)
        np.testing.assert_array_almost_equal(y_complete, gr_y_complete, decimal = 2)
        np.testing.assert_array_almost_equal(x_new, gr_x_new, decimal = 2)
        np.testing.assert_array_almost_equal(y_new, gr_y_new, decimal = 2)
        
        