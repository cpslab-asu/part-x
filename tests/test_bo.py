import pickle
import numpy as np
import pytest
import pathlib

from numpy import random

from partx.gpr import InternalGPR
from partx.utilities.utils import Fn, compute_robustness
from partx.utilities.sampling import uniform_sampling
from partx.bo import InternalBO
from partx.utilities.utils import OracleCreator


@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)

@pytest.fixture()
def oracle_info() -> OracleCreator:
 return OracleCreator(None, 1,1)

@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
    

def test_internalBO(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    
    region_support = np.array([[-1, 1]])
    gpr_model = InternalGPR()
    bo = InternalBO()

    with open(data_path / "test_1_internalBO.pickle", "rb") as f:
        # pickle.dump((in_samples_1, out_samples_1, x_new), f)
        in_samples_1, out_samples_1, gr_x_new = pickle.load(f)
    
    x_new = bo.sample(
        in_samples_1, out_samples_1, region_support, gpr_model, oracle_info, rng
    )
    
    np.testing.assert_array_almost_equal(x_new, gr_x_new, decimal = 3)
        

def test2_internalBO(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    # def internal_function(X):
    #     return X[0] ** 2 + X[1] ** 2
    #     # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

    
    region_support = np.array([[-1, 1], [-1, 1]])

    # func1 = Fn(internal_function)
    # in_samples_1 = uniform_sampling(20, region_support, 2, oracle_info, rng)
    # out_samples_1 = compute_robustness(in_samples_1, func1)

    gpr_model = InternalGPR()
    bo = InternalBO()
    with open(data_path / "test_2_internalBO.pickle", "rb") as f:
        # pickle.dump((in_samples_1, out_samples_1, x_new), f)
        in_samples_1, out_samples_1, gr_x_new = pickle.load(f)

    x_new = bo.sample(
        in_samples_1, out_samples_1, region_support, gpr_model, oracle_info, rng
    )

    

    # print(aser)
    np.testing.assert_array_almost_equal(x_new, gr_x_new, decimal = 1)


def test3_internalBO(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
    gpr_model = InternalGPR()
    bo = InternalBO()

    with open(data_path / "test_3_internalBO.pickle", "rb") as f:
        # pickle.dump((in_samples_1, out_samples_1, x_new), f)
        in_samples_1, out_samples_1, gr_x_new = pickle.load(f)

    x_new = bo.sample(
        in_samples_1, out_samples_1, region_support, gpr_model, oracle_info, rng
    )

    # with open(data_path / "test_3_internalBO.pickle", "wb") as f:
        # pickle.dump((in_samples_1, out_samples_1, x_new), f)
        # in_samples_1, out_samples_1, gr_x_new = pickle.load(f)
    
    np.testing.assert_array_almost_equal(x_new, gr_x_new, decimal = 1)



