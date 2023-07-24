import numpy as np
import unittest
import pathlib
import time 
import pickle

from partx.coreAlgorithm import PartXOptions, run_single_replication
from partx.gprInterface import InternalGPR
from partx.bayesianOptimization import InternalBO
from partx.results import fv_without_gp
from partx.utils import OracleCreator

oracle_func = None

class TestSingleReplication(unittest.TestCase):
    def test1_single_replication(self):
        BENCHMARK_NAME = "Testing_t1"
        init_reg_sup = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2
        max_budget = 100
        init_budget = 10
        bo_budget = 10
        cs_budget = 10
        alpha = 0.05
        R = 20
        M = 500
        delta = 0.001
        fv_quantiles_for_gp = [0.01, 0.05, 0.5]
        branching_factor = 2
        uniform_partitioning = True
        start_seed = 123
        gpr_model = InternalGPR()
        bo_model = InternalBO()
        oracle_info = OracleCreator(oracle_func, 1,1)
        options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, "lhs_sampling", "lhs_sampling", "lhs_sampling")

        

        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

        
        inputs = [0, options, internal_function, oracle_info, pathlib.Path("tests/coreAlgorithm/test")]
        t = time.time()
        run_single_replication(inputs)
        
        # print(f"Total Time Taken = {time.time() - t}")


    def test2_single_replication(self):
        BENCHMARK_NAME = "Testing_t2"
        init_reg_sup = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2
        max_budget = 100
        init_budget = 10
        bo_budget = 10
        cs_budget = 10
        alpha = 0.05
        R = 20
        M = 500
        delta = 0.001
        fv_quantiles_for_gp = [0.01, 0.05, 0.5]
        branching_factor = 2
        uniform_partitioning = True
        start_seed = 123
        gpr_model = InternalGPR()
        bo_model = InternalBO()
        oracle_info = OracleCreator(oracle_func, 1,1)
        options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, "uniform_sampling", "uniform_sampling", "uniform_sampling")

        

        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

        
        inputs = [0, options, internal_function, oracle_info, pathlib.Path("tests/coreAlgorithm/test")]
        t = time.time()
        run_single_replication(inputs)
        # print(f"Total Time Taken = {time.time() - t}")


    def test3_single_replication(self):
        BENCHMARK_NAME = "Testing_t3"
        init_reg_sup = np.array([[-1., 1.], [-1., 1.]])
        tf_dim = 2
        max_budget = 1000
        init_budget = 10
        bo_budget = 10
        cs_budget = 10
        alpha = 0.05
        R = 20
        M = 500
        delta = 0.001
        fv_quantiles_for_gp = [0.01, 0.05, 0.5]
        branching_factor = 2
        uniform_partitioning = True
        start_seed = 123
        gpr_model = InternalGPR()
        bo_model = InternalBO()
        oracle_info = OracleCreator(oracle_func, 1,1)
        options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, "uniform_sampling", "uniform_sampling", "uniform_sampling")

        

        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

        
        inputs = [0, options, internal_function, oracle_info, pathlib.Path("tests/coreAlgorithm/test")]
        t = time.time()
        run_single_replication(inputs)
        with open("tests/coreAlgorithm/test/Testing_t3_result_generating_files/Testing_t3_0.pkl", "rb") as f:
            
            ftree = pickle.load(f)

        rng = np.random.default_rng(12345)
        a,b = fv_without_gp(ftree, options)

    def test4_single_replication(self):
        BENCHMARK_NAME = "Testing_t4"
        init_reg_sup = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        tf_dim = 2
        max_budget = 100
        init_budget = 10
        bo_budget = 10
        cs_budget = 10
        alpha = 0.05
        R = 20
        M = 500
        delta = 0.05
        fv_quantiles_for_gp = [0.01, 0.05, 0.5]
        branching_factor = 2
        uniform_partitioning = True
        start_seed = 123
        gpr_model = InternalGPR()
        bo_model = InternalBO()
        oracle_info = OracleCreator(oracle_func, 1,1)
        options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, "uniform_sampling", "uniform_sampling", "uniform_sampling")

        

        def internal_function(X):
            return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

        
        inputs = [0, options, internal_function, oracle_info, pathlib.Path("tests/coreAlgorithm/test")]
        t = time.time()
        run_single_replication(inputs)
        with open("tests/coreAlgorithm/test/Testing_t4_result_generating_files/Testing_t4_0.pkl", "rb") as f:
            ftree = pickle.load(f)

        rng = np.random.default_rng(12345)
        a,b = fv_without_gp(ftree, options)