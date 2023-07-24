import pickle
import numpy as np
import unittest
from partx.quantileClassification.regionQuantileEstimation import mc_step, estimate_mc, estimate_quantiles
from partx.sampling import uniform_sampling
from partx.utils import Fn, compute_robustness, OracleCreator
from partx.gprInterface import InternalGPR
# from partx.coreAlgorithm import OracleCreator

oracle_func = None

class TestQuantileEstimation(unittest.TestCase):
    def test1_mc_step(self):
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
        alpha = 0.05

        min_quantile, max_quantile = mc_step(x_train, y_train, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type = "lhs_sampling")

        with open("./tests/quantileClassification/goldResources/mc_step_1_minquantile.pickle", "rb") as f:
            # pickle.dump(min_quantile,f)
            min_q_gr = pickle.load(f)

        with open("./tests/quantileClassification/goldResources/mc_step_1_maxquantile.pickle", "rb") as f:
            # pickle.dump(max_quantile,f)
            max_q_gr = pickle.load(f)
        
        np.testing.assert_almost_equal(min_quantile, min_q_gr, 1)
        np.testing.assert_almost_equal(max_quantile, max_q_gr, 1)

    def test2_mc_step(self):
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

        min_quantile, max_quantile = mc_step(x_train, y_train, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type = "uniform_sampling")

        with open("./tests/quantileClassification/goldResources/mc_step_1_ufs_minquantile.pickle", "rb") as f:
            # pickle.dump(min_quantile,f)
            min_q_gr = pickle.load(f)

        with open("./tests/quantileClassification/goldResources/mc_step_1_ufs_maxquantile.pickle", "rb") as f:
            # pickle.dump(max_quantile,f)
            max_q_gr = pickle.load(f)
        
        np.testing.assert_almost_equal(min_quantile, min_q_gr, 1)
        np.testing.assert_almost_equal(max_quantile, max_q_gr, 1)

    def test3_estimateMC(self):
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

        R = 100
        M = 10000
        gpr_model = InternalGPR()
        x_train = uniform_sampling(100, region_support, tf_dim, oracle_info, rng)
        y_train = compute_robustness(x_train, tf)
        alpha = 0.05

        min_delta_quantile, max_delta_quantile = estimate_quantiles(x_train, y_train, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type = "lhs_sampling")
        with open("./tests/quantileClassification/goldResources/mc_quantile.pickle", "rb")  as f:
            # pickle.dump([min_delta_quantile, max_delta_quantile], f)
            gr_min_delta_quantile, gr_max_delta_quantile = pickle.load(f)

        assert min_delta_quantile == gr_min_delta_quantile
        assert max_delta_quantile == gr_max_delta_quantile
