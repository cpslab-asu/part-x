import pickle
import numpy as np
import pickle
import unittest

from partx.utils import Fn, compute_robustness
from partx.sampling import uniform_sampling, lhs_sampling
from partx.gprInterface import (
    GPR,
    GaussianProcessRegressorStructure,
    InternalGPR,
)

from partx.utils import OracleCreator

oracle_func = None


class Test_GPR(unittest.TestCase):
    def test1_GPR(self):
        gpr_model = InternalGPR()
        gpr = GPR(gpr_model)
        oracle_info = OracleCreator(oracle_func,1,1)
        def internal_function(X):
            return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        rng = np.random.default_rng(12345)
        region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])

        func1 = Fn(internal_function)
        in_samples_1 = uniform_sampling(20, region_support, 3, oracle_info, rng)
        out_samples_1 = compute_robustness(in_samples_1, func1)
        # gpr.fit(in_samples_1, out_samples_1)

        func2 = Fn(internal_function)
        in_samples_2 = uniform_sampling(1, region_support, 3, oracle_info, rng)
        out_samples_2 = compute_robustness(in_samples_2, func2)

        self.assertRaises(TypeError, gpr.fit,  np.array([in_samples_1]), out_samples_1)
        self.assertRaises(TypeError, gpr.fit, in_samples_1, np.array([out_samples_1]).T)
        self.assertRaises(TypeError, gpr.fit, in_samples_1, out_samples_2)
        gpr.fit(in_samples_1, out_samples_1)

        self.assertRaises(TypeError, gpr.predict, np.array([in_samples_1]))

        y_pred_1, y_std_1 = gpr.predict(in_samples_1)
        y_pred_2, y_std_2 = gpr.predict(in_samples_2)

        with open("tests/gprInterface/goldResources/test_1_gpr.pickle", "rb") as f:
            # pickle.dump([y_pred_1, y_std_1], f)
            gr_pred_1, gr_std_1 = pickle.load(f)
        
        with open("tests/gprInterface/goldResources/test_2_gpr.pickle", "rb") as f:
            # pickle.dump([y_pred_2, y_std_2], f)
            gr_pred_2, gr_std_2 = pickle.load(f)

        np.testing.assert_array_almost_equal(y_pred_1, gr_pred_1, decimal = 2)
        np.testing.assert_array_almost_equal(y_std_1, gr_std_1, decimal = 2)
        np.testing.assert_array_almost_equal(y_pred_2, gr_pred_2, decimal = 2)
        np.testing.assert_array_almost_equal(y_std_2, gr_std_2, decimal = 2)

if __name__ == "__main__":
    unittest.main()
