import pickle
import numpy as np
import unittest

from partx.sampling import uniform_sampling
from partx.utils import Fn, compute_robustness, calculate_volume
from partx.quantileClassification import estimate_quantiles, classification
from partx.gprInterface import InternalGPR
from partx.utils import OracleCreator

oracle_func = None

class TestClassification(unittest.TestCase):

    def test1_Classification(self):
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

        R = 20
        M = 500
        gpr_model = InternalGPR()
        x_train = uniform_sampling(100, region_support, tf_dim, oracle_info, rng)
        y_train = compute_robustness(x_train, tf)
        alpha = 0.05

        
        min_delta_quantile, max_delta_quantile = estimate_quantiles(x_train, y_train, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type = "lhs_sampling")
        region_class = "r"
        delta = 0.01
        min_volume = (delta ** tf_dim) * calculate_volume(region_support)
        out = classification(region_support, region_class, min_volume, min_delta_quantile, max_delta_quantile)
        # print(min_delta_quantile, max_delta_quantile)
        # print(out)

        
