import pickle
import numpy as np
import pickle
import unittest

from partx.sampling import lhs_sampling, uniform_sampling, OOBError
from partx.utils import OracleCreator

def oracle_func(X):
    return X[0]**2 + X[1]**2 - 0.25

def oracle_func_3d(X):
    return X[0]**2 + X[1]**2 + X[2]**2 - 0.25

def oracle_func_4d(X):
    return X[0]**2 + X[1]**2 + X[2]**2 + X[3] - 0.25

class TestSamplingWithConstraints(unittest.TestCase):
    def test1_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func, 100,1)

        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 3
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
    
    def test2_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func_3d, 100,1)
        region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng))

    def test3_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func, 100,1)
        region_support = np.array([[1, -1], [1, -1]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
    
    def test4_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func, 100,1)
        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_unif = uniform_sampling(
            num_samples, region_support, tf_dim, oracle_info, rng
        )
        
        with open("./tests/sampling/goldResources/unif_samp_wcons_t1.pickle", "rb") as f:
            # pickle.dump(samples_in_unif, f)
            gr = pickle.load(f)

        np.testing.assert_array_equal(samples_in_unif, gr)        

    def test5_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func_4d, 100,1)
        region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
        tf_dim = 4
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_unif = uniform_sampling(
            num_samples, region_support, tf_dim, oracle_info, rng
        )
        
        with open("./tests/sampling/goldResources/unif_samp_wcons_t2.pickle", "rb") as f:
            # pickle.dump(samples_in_unif, f)
            gr = pickle.load(f)
        
        np.testing.assert_array_equal(samples_in_unif, gr)

    def test5_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func_4d, 2,1)
        region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
        tf_dim = 4
        num_samples = 1000
        seed = 10001
        rng = np.random.default_rng(seed)
        
        self.assertRaises(OOBError, lambda:uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
        

    def test1_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 100,1)
        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 3
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
    
    def test2_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func_3d, 100,1)
        region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng))

    def test3_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 100,1)
        region_support = np.array([[1, -1], [1, 1]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
    
    def test4_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 100,1)
        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_lhs = lhs_sampling(
            num_samples, region_support, tf_dim, oracle_info, rng
        )
        
        with open("./tests/sampling/goldResources/lhs_samp_wcons_t1.pickle", "rb") as f:
            # pickle.dump(samples_in_lhs, f)
            gr = pickle.load(f)
        
        np.testing.assert_array_equal(samples_in_lhs, gr)

    def test5_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func_4d, 1000,1)
        region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
        tf_dim = 4
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_lhs = lhs_sampling(
            num_samples, region_support, tf_dim, oracle_info, rng
        )
        
        with open("./tests/sampling/goldResources/lhs_samp_wcons_t2.pickle", "rb") as f:
            # pickle.dump(samples_in_lhs, f)
            gr = pickle.load(f)
        
        np.testing.assert_array_equal(samples_in_lhs, gr)

    def test6_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 2,1)
        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 2
        num_samples = 10000
        seed = 10001
        rng = np.random.default_rng(seed)
        
        self.assertRaises(OOBError, lambda:lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng))

if __name__ == "__main__":
    unittest.main()
