import pickle
import numpy as np
import unittest

from partx.sampling import uniform_sampling
from partx.utils import Fn, compute_robustness
from partx.numerical import calculate_mc_integral
from partx.gprInterface import InternalGPR
from partx.utils import OracleCreator

oracle_func = None

class TestCalculateMCIntegral(unittest.TestCase):

    def test1_cal_mc_integral(self):
        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

        oracle_info = OracleCreator(oracle_func, 1,1)
        tf = Fn(internal_function)
        rng = np.random.default_rng(12345)
        region_support = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2

        R = 20
        M = 500
        gpr_model = InternalGPR()
        x_train = uniform_sampling(100, region_support, tf_dim, oracle_info, rng)
        y_train = compute_robustness(x_train, tf)
        
        integral = calculate_mc_integral(x_train, y_train, region_support, tf_dim, R, M, gpr_model, oracle_info, rng, sampling_type="lhs_sampling")
        with open("./tests/numerical/goldResources/integral_1.pickle", "rb") as f:
            # pickle.dump(integral,f)
            integral_gold = pickle.load(f)
        
        np.testing.assert_almost_equal(integral, integral_gold, 1)

    def test2_cal_mc_integral(self):
        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50
        oracle_info = OracleCreator(oracle_func, 1,1)
        tf = Fn(internal_function)
        rng = np.random.default_rng(12345)
        region_support = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2

        R = 20
        M = 500
        gpr_model = InternalGPR()
        x_train = uniform_sampling(100, region_support, tf_dim, oracle_info, rng)
        y_train = compute_robustness(x_train, tf)
        
        integral = calculate_mc_integral(x_train, y_train, region_support, tf_dim, R, M, gpr_model, oracle_info, rng, sampling_type="uniform_sampling")
        with open("./tests/numerical/goldResources/integral_2.pickle", "rb") as f:
            # pickle.dump(integral,f)
            integral_gold = pickle.load(f)
        
        np.testing.assert_almost_equal(integral, integral_gold, 1)
        
    def test3_cal_mc_integral(self):
        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

        tf = Fn(internal_function)
        oracle_info = OracleCreator(oracle_func, 1,1)
        rng = np.random.default_rng(12345)
        region_support = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2

        R = 3
        M = 10
        gpr_model = InternalGPR()
        x_train = uniform_sampling(100, region_support, tf_dim, oracle_info, rng)
        y_train = compute_robustness(x_train, tf)
        
        integral = calculate_mc_integral(x_train, y_train, region_support, tf_dim, R, M, gpr_model, oracle_info, rng, sampling_type="uniform_sampling")
        with open("./tests/numerical/goldResources/integral_3.pickle", "rb") as f:
            # pickle.dump(integral,f)
            integral_gold = pickle.load(f)
        
        np.testing.assert_almost_equal(integral, integral_gold, 1)