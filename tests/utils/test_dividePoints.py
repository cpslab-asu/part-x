import numpy as np
import unittest
import pickle

from partx.sampling import uniform_sampling, lhs_sampling
from partx.utils import branch_region, compute_robustness, Fn, divide_points, OracleCreator
from partx.utils.pointInSubRegion import testPointInSubRegion
# from partx.coreAlgorithm import OracleCreator

oracle_func = None

class TestDividePoints(unittest.TestCase):
    def test1_divide_points(self):
        def test_function(X):
            return X[0] ** 2 + X[1] ** 2
        oracle_info = OracleCreator(oracle_func, 1,1)
        funct = Fn(test_function)
        
        region_support = np.array([[-1., 1.], [-1., 2.]])
        rng = np.random.default_rng(12345)
        samples = lhs_sampling(1000, region_support, 2, oracle_info, rng)
        samples_out = compute_robustness(samples, funct)

        reg_sup = branch_region(region_support, 1, False, 4, rng)
        x, y = divide_points(samples, samples_out, reg_sup)
        
        with open("./tests/utils/goldResources/divide_points_t1_x.pickle", "rb") as f:
            gr_x = pickle.load(f)
            # pickle.dump(x, f)

        with open("./tests/utils/goldResources/divide_points_t1_y.pickle", "rb") as f:
            gr_y = pickle.load(f)
            # pickle.dump(y, f)

        for _x, _y, _gr_x, _gr_y in zip(x,y, gr_x, gr_y):
            np.testing.assert_array_equal(_x, _gr_x)
            np.testing.assert_array_equal(_y, _gr_y)

    def test2_divide_points(self):
        def test_function(X):
            return X[0] ** 2 + X[1] ** 2
        
        funct = Fn(test_function)
        rng = np.random.default_rng(12345)
        region_support = np.array([[-1., 1.], [-1., 2.]])
        
        samples = np.array([[]])
        samples_out = np.array([])

        reg_sup = branch_region(region_support, 1, False, 4, rng)
        x, y = divide_points(samples, samples_out, reg_sup)
        
        for _x, _y in zip(x,y):
            assert _x.shape == (1,0)
            assert _y.shape == (0,)

