import pickle
import numpy as np
import pytest
import pathlib

from numpy import random
from partx.utilities.sampling import uniform_sampling
from partx.utilities.utils import Fn, compute_robustness, OracleCreator
from partx.utilities.stat_utils import calculate_mc_integral
from partx.gpr import InternalGPR

@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)

@pytest.fixture()
def oracle_info() -> OracleCreator:
 return OracleCreator(None, 1,1)

@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
    

# def test1_cal_mc_integral(rng: random.Generator, oracle_info: OracleCreator):
#     def internal_function(X):
#         return (5*X[0]**2-4)

#     tf = Fn(internal_function)
#     region_support = np.array([[-10.,10.]])
#     tf_dim = 1

#     R = 10
#     M = 20
#     gpr_model = InternalGPR()
#     x_train = uniform_sampling(500, region_support, tf_dim, oracle_info, rng)
#     y_train = compute_robustness(x_train, tf)
    
#     integral = calculate_mc_integral(x_train, y_train, region_support, tf_dim, R, M, gpr_model, oracle_info, rng, sampling_type="lhs_sampling")
    
#     print(integral)
#     print(dvsa)

def test1_cal_mc_integral(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    
    with open(data_path / "integral_test_1_data.pickle", "rb") as f:
        # pickle.dump((x_train, y_train),f)
        x_train, y_train = pickle.load(f)

    region_support = np.array([[-1., 1.], [-1., 1.]])
    tf_dim = 2

    R = 20
    M = 1000
    gpr_model = InternalGPR()

    integral = calculate_mc_integral(x_train, y_train, region_support, tf_dim, R, M, gpr_model, oracle_info, rng, sampling_type="lhs_sampling")
    np.testing.assert_almost_equal(integral, 0.468, 2)

def test2_cal_mc_integral(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    
    with open(data_path / "integral_test_1_data.pickle", "rb") as f:
        # pickle.dump((x_train, y_train),f)
        x_train, y_train = pickle.load(f)

    region_support = np.array([[-1., 1.], [-1., 1.]])
    tf_dim = 2

    R = 1
    M = 1000
    gpr_model = InternalGPR()

    integral = calculate_mc_integral(x_train, y_train, region_support, tf_dim, R, M, gpr_model, oracle_info, rng, sampling_type="lhs_sampling")
    np.testing.assert_almost_equal(integral, 0.468, 1)