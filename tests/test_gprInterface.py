import pickle
import numpy as np
import numpy.random as random
import pickle
import pytest
import pathlib


from partx.utilities.sampling import uniform_sampling, lhs_sampling
from partx.utilities.utils import OracleCreator

from partx.gpr import (
    GPR,
    GPRSkeleton,
    InternalGPR,
)

@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)

@pytest.fixture()
def oracle_info() -> OracleCreator:
 return OracleCreator(None, 1,1)

@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
    

def test_GPR_incorrect_input_shape_fitting(rng: random.Generator, oracle_info: OracleCreator):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    def internal_function(X):
        return X[:,0] ** 2 + X[:,1] ** 2 + X[:,2] ** 2

    
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])

    in_samples_1 = uniform_sampling(20, region_support, 3, oracle_info, rng)
    out_samples_1 = internal_function(in_samples_1)
    
    
    with pytest.raises(TypeError):
        gpr.fit(np.array([in_samples_1]), out_samples_1)

def test_GPR_incorrect_output_shape_fitting(rng: random.Generator, oracle_info: OracleCreator):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    def internal_function(X):
        return X[:,0] ** 2 + X[:,1] ** 2 + X[:,2] ** 2

    
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])

    in_samples_1 = uniform_sampling(20, region_support, 3, oracle_info, rng)
    out_samples_1 = internal_function(in_samples_1)
    
    
    with pytest.raises(TypeError):
        gpr.fit(in_samples_1, np.array([out_samples_1]).T)

def test_GPR_inconsistent_iodat_fitting(rng: random.Generator, oracle_info: OracleCreator):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    def internal_function(X):
        return X[:,0] ** 2 + X[:,1] ** 2 + X[:,2] ** 2

    
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])

    in_samples_1 = uniform_sampling(20, region_support, 3, oracle_info, rng)
    out_samples_1 = internal_function(in_samples_1)

    in_samples_2 = uniform_sampling(10, region_support, 3, oracle_info, rng)
    out_samples_2 = internal_function(in_samples_2)
    
    
    with pytest.raises(TypeError):
        gpr.fit(in_samples_1, out_samples_2)

def test_GPR_inconsistent_input_prediction(rng: random.Generator, oracle_info: OracleCreator):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    def internal_function(X):
        return X[:,0] ** 2 + X[:,1] ** 2 + X[:,2] ** 2

    
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])

    in_samples_1 = uniform_sampling(20, region_support, 3, oracle_info, rng)
    out_samples_1 = internal_function(in_samples_1)

    in_samples_2 = uniform_sampling(10, region_support, 3, oracle_info, rng)
    out_samples_2 = internal_function(in_samples_2)
    
    gpr.fit(in_samples_1, out_samples_1)

    with pytest.raises(TypeError):
        gpr.predict(np.array([in_samples_1]))

def test_GPR_output_prediction(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    def internal_function(X):
        return X[:,0] ** 2 + X[:,1] ** 2 + X[:,2] ** 2

    
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])

    in_samples_1 = uniform_sampling(20, region_support, 3, oracle_info, rng)
    out_samples_1 = internal_function(in_samples_1)

    in_samples_2 = uniform_sampling(10, region_support, 3, oracle_info, rng)
    out_samples_2 = internal_function(in_samples_2)
    
    gpr.fit(in_samples_1, out_samples_1)

    y_pred_1, y_std_1 = gpr.predict(in_samples_1)
    y_pred_2, y_std_2 = gpr.predict(in_samples_2)
    

    with open(data_path / "test_1_gpr.pickle", "rb") as f:
        # pickle.dump([y_pred_1, y_std_1], f)
        gr_pred_1, gr_std_1 = pickle.load(f)

    with open(data_path / "test_2_gpr.pickle", "rb") as f:
        # pickle.dump([y_pred_2, y_std_2], f)
        gr_pred_2, gr_std_2 = pickle.load(f)


    np.testing.assert_array_almost_equal(y_pred_1, gr_pred_1, decimal = 2)
    np.testing.assert_array_almost_equal(y_std_1, gr_std_1, decimal = 2)
    np.testing.assert_array_almost_equal(y_pred_2, gr_pred_2, decimal = 2)
    np.testing.assert_array_almost_equal(y_std_2, gr_std_2, decimal = 2)