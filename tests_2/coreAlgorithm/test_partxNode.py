import numpy as np
import unittest
from partx.coreAlgorithm import PartXNode, PartXOptions
from partx.gprInterface import InternalGPR
from partx.bayesianOptimization import InternalBO
from partx.utils import Fn, compute_robustness
from partx.sampling import uniform_sampling
from partx.utils import OracleCreator

oracle_func = None

class TestPartXNode(unittest.TestCase):
    def test1_unclassified_regions(self):
        BENCHMARK_NAME = "Testing"
        init_reg_sup = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2
        max_budget = 100
        init_budget = 10
        bo_budget = 10
        cs_budget = 20
        alpha = 0.05
        R = 20
        M = 500
        delta = 0.001
        fv_quantiles_for_gp = [0.01, 0.05, 0.5]
        branching_factor = 2
        uniform_partitioning = "True"
        start_seed = 12345
        gpr_model = InternalGPR()
        bo_model = InternalBO()
        oracle_info = OracleCreator(oracle_func, 1,1)
        options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model)

        region_support = np.array([[-1., 1.], [-1., 1.]])

        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50
        

        testFunction = Fn(internal_function)
        samples_in = np.array([[]])
        samples_out = np.array([])

        rng = np.random.default_rng(12345)

        node = PartXNode(0, 0, region_support, samples_in, samples_out, 0, region_class="r")
        new_class = node.samples_management_unclassified(testFunction, options, oracle_info, rng)
        assert new_class == "r"
        assert node.new_region_class == "r"
        assert node.region_class == "r"

    def test2_unclassified_regions(self):
        BENCHMARK_NAME = "Testing"
        init_reg_sup = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2
        max_budget = 100
        init_budget = 10
        bo_budget = 10
        cs_budget = 20
        alpha = 0.05
        R = 20
        M = 500
        delta = 0.001
        fv_quantiles_for_gp = [0.01, 0.05, 0.5]
        branching_factor = 2
        uniform_partitioning = "True"
        start_seed = 12345
        gpr_model = InternalGPR()
        bo_model = InternalBO()
        oracle_info = OracleCreator(oracle_func, 1,1)
        
        options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model)

        region_support = np.array([[-1., 1.], [-1., 1.]])

        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50
        
        testFunction = Fn(internal_function)
        rng = np.random.default_rng(12345)
        samples_in = uniform_sampling(8, region_support, tf_dim, oracle_info, rng)
        samples_out = compute_robustness(samples_in, testFunction)

        

        node = PartXNode(0, 0, region_support, samples_in, samples_out, 0, region_class="r")
        new_class = node.samples_management_unclassified(testFunction, options, oracle_info, rng)
        assert new_class == "r"
        assert node.new_region_class == "r"
        assert node.region_class == "r"
        


    def test3_classified_regions(self):
        BENCHMARK_NAME = "Testing"
        init_reg_sup = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2
        max_budget = 100
        init_budget = 10
        bo_budget = 10
        cs_budget = 20
        alpha = 0.05
        R = 20
        M = 500
        delta = 0.001
        fv_quantiles_for_gp = [0.01, 0.05, 0.5]
        branching_factor = 2
        uniform_partitioning = "True"
        start_seed = 12345
        gpr_model = InternalGPR()
        bo_model = InternalBO()
        oracle_info = OracleCreator(oracle_func, 1,1)

        options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model)

        region_support = np.array([[-1., 1.], [-1., 1.]])

        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50
        
        testFunction = Fn(internal_function)
        rng = np.random.default_rng(12345)
        samples_in = uniform_sampling(30, region_support, tf_dim, oracle_info, rng)
        samples_out = compute_robustness(samples_in, testFunction)

        

        node = PartXNode(0, 0, region_support, samples_in, samples_out, 0, region_class="+")
        new_class = node.samples_management_classified(30, testFunction, options, oracle_info, rng)
        assert new_class == "r+"
        assert node.new_region_class == "r+"
        assert node.region_class == "r+"


    def test4_classified_regions(self):
        BENCHMARK_NAME = "Testing"
        init_reg_sup = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2
        max_budget = 100
        init_budget = 10
        bo_budget = 10
        cs_budget = 20
        alpha = 0.05
        R = 20
        M = 500
        delta = 0.001
        fv_quantiles_for_gp = [0.01, 0.05, 0.5]
        branching_factor = 2
        uniform_partitioning = "True"
        start_seed = 12345
        gpr_model = InternalGPR()
        bo_model = InternalBO()
        oracle_info = OracleCreator(oracle_func, 1,1)
        
        options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model)

        region_support = np.array([[-1., 1.], [-1., 1.]])

        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50
        
        testFunction = Fn(internal_function)
        rng = np.random.default_rng(12345)
        samples_in = uniform_sampling(30, region_support, tf_dim, oracle_info, rng)
        samples_out = compute_robustness(samples_in, testFunction)

        

        node = PartXNode(0, 0, region_support, samples_in, samples_out, 0, region_class="-")
        new_class = node.samples_management_classified(30, testFunction, options, oracle_info, rng)
        assert new_class == "r-"
        assert node.new_region_class == "r-"
        assert node.region_class == "r-"

        