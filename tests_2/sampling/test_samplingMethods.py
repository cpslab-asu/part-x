import pickle
import numpy as np
import pickle
import unittest

from partx.sampling import lhs_sampling, uniform_sampling
from partx.utils import OracleCreator

oracle_func = None

class TestSampling(unittest.TestCase):
    def test1_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)

        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 3
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
    
    def test2_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng))

    def test3_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[1, -1], [1, -1]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
    
    def test4_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_unif = uniform_sampling(
            num_samples, region_support, tf_dim, oracle_info, rng
        )
        
        with open("./tests/sampling/goldResources/unif_samp_t1.pickle", "rb") as f:
            # pickle.dump(samples_in_unif, f)
            gr = pickle.load(f)
        
        np.testing.assert_array_equal(samples_in_unif, gr)

    def test5_uniform_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
        tf_dim = 4
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_unif = uniform_sampling(
            num_samples, region_support, tf_dim, oracle_info, rng
        )
        
        with open("./tests/sampling/goldResources/unif_samp_t2.pickle", "rb") as f:
            # pickle.dump(samples_in_unif, f)
            gr = pickle.load(f)
        
        np.testing.assert_array_equal(samples_in_unif, gr)

    def test1_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 3
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
    
    def test2_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng))

    def test3_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[1, -1], [1, 1]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        self.assertRaises(ValueError, lambda:lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng))
    
    def test4_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[-1, 1], [-1, 1]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_lhs = lhs_sampling(
            num_samples, region_support, tf_dim, oracle_info, rng
        )
        
        with open("./tests/sampling/goldResources/lhs_samp_t1.pickle", "rb") as f:
            # pickle.dump(samples_in_lhs, f)
            gr = pickle.load(f)
        
        np.testing.assert_array_equal(samples_in_lhs, gr)

    def test5_lhs_sampling(self):
        oracle_info = OracleCreator(oracle_func, 1,1)
        region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
        tf_dim = 4
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_lhs = lhs_sampling(
            num_samples, region_support, tf_dim, oracle_info, rng
        )
        
        with open("./tests/sampling/goldResources/lhs_samp_t2.pickle", "rb") as f:
            # pickle.dump(samples_in_lhs, f)
            gr = pickle.load(f)
        
        np.testing.assert_array_equal(samples_in_lhs, gr)

if __name__ == "__main__":
    unittest.main()
