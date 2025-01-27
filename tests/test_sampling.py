import pathlib
import pickle

import pytest
import numpy as np
import numpy.random as random

# from pa.sampling import lhs_sampling, uniform_sampling
from partx.utilities.sampling import lhs_sampling, uniform_sampling, OOBError
from partx.utilities.utils import OracleCreator

def oracle_func_1d(X):
    return X[0]**2 + X[1]**2 - 0.25

def oracle_func_3d(X):
    return X[0]**2 + X[1]**2 + X[2]**2 - 0.25

def oracle_func_4d(X):
    return X[0]**2 + X[1]**2 + X[2]**2 + X[3] - 0.25


@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)

@pytest.fixture()
def oracle_info() -> OracleCreator:
 return OracleCreator(None, 1,1)

@pytest.fixture()
def oracle_info_1() -> OracleCreator:
    return OracleCreator(oracle_func_1d, 100,1)

@pytest.fixture()
def oracle_info_3() -> OracleCreator:
    return OracleCreator(oracle_func_3d, 100,1)

@pytest.fixture()
def oracle_info_4() -> OracleCreator:
    return OracleCreator(oracle_func_4d, 100,1)

@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
    

def test_uniform_sampling_2d_region_3d_tf(rng: random.Generator, oracle_info: OracleCreator):
    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 3
    num_samples = 10

    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng)


def test_uniform_sampling_3d_region_2d_tf(rng: random.Generator, oracle_info: OracleCreator):
    region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
    tf_dim = 2
    num_samples = 10

    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng)


def test_uniform_sampling_region_desc(rng: random.Generator, oracle_info: OracleCreator):
    region_support = np.array([[1, -1], [1, -1]])
    tf_dim = 2
    num_samples = 10
    
    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, oracle_info, rng)

def test_uniform_sampling_array_shape_check_2dim(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 2
    num_samples = 10

    samples_in_unif = uniform_sampling(
        num_samples, region_support, tf_dim, oracle_info, rng
    )
    
    with open(data_path / "unif_samp_t1.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)

def test_uniform_sampling_array_shape_check_4dim(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
    tf_dim = 4
    num_samples = 10

    samples_in_unif = uniform_sampling(
        num_samples, region_support, tf_dim, oracle_info, rng
    )
    
    with open(data_path / "unif_samp_t2.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)

def test_lhs_sampling_2d_region_3d_tf(rng: random.Generator, oracle_info: OracleCreator):
    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 3
    num_samples = 10

    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng)


def test_lhs_sampling_3d_region_2d_tf(rng: random.Generator, oracle_info: OracleCreator):
    region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
    tf_dim = 2
    num_samples = 10

    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng)


def test_lhs_sampling_region_desc(rng: random.Generator, oracle_info: OracleCreator):
    region_support = np.array([[1, -1], [1, -1]])
    tf_dim = 2
    num_samples = 10
    
    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, oracle_info, rng)

def test_lhs_sampling_array_shape_check_2dim(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 2
    num_samples = 10

    samples_in_unif = lhs_sampling(
        num_samples, region_support, tf_dim, oracle_info, rng
    )
    
    with open(data_path / "lhs_samp_t1.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)

def test_lhs_sampling_array_shape_check_4dim(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
    tf_dim = 4
    num_samples = 10

    samples_in_unif = lhs_sampling(
        num_samples, region_support, tf_dim, oracle_info, rng
    )
    
    with open(data_path / "lhs_samp_t2.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)


def test_uniform_sampling_2d_region_3d_tf_wcons(rng: random.Generator, oracle_info_1: OracleCreator):

    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 3
    num_samples = 10

    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, oracle_info_1, rng)
 
def test_uniform_sampling_3d_region_2d_tf_wcons(rng: random.Generator, oracle_info_3: OracleCreator):

    region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
    tf_dim = 2
    num_samples = 10

    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, oracle_info_3, rng)

def test_uniform_sampling_region_desc_wcons(rng: random.Generator, oracle_info_1: OracleCreator):

    region_support = np.array([[1, -1], [1, -1]])
    tf_dim = 2
    num_samples = 10

    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, oracle_info_1, rng)
    
def test_uniform_sampling_array_shape_check_2dim_wcons(rng: random.Generator, oracle_info_1: OracleCreator, data_path: pathlib.Path):

    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 2
    num_samples = 10

    samples_in_unif = uniform_sampling(
        num_samples, region_support, tf_dim, oracle_info_1, rng
    )
    
    with open(data_path / "unif_samp_wcons_t1.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)

    np.testing.assert_array_equal(samples_in_unif, gr)       

def test_uniform_sampling_array_shape_check_4dim_wcons(rng: random.Generator, oracle_info_4: OracleCreator, data_path: pathlib.Path):
        # oracle_info = OracleCreator(oracle_func_4d, 100,1)
        region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
        tf_dim = 4
        num_samples = 10

        samples_in_unif = uniform_sampling(
            num_samples, region_support, tf_dim, oracle_info_4, rng
        )
        
        with open(data_path / "unif_samp_wcons_t2.pickle", "rb") as f:
            # pickle.dump(samples_in_unif, f)
            gr = pickle.load(f)
        
        np.testing.assert_array_equal(samples_in_unif, gr)



def test_lhs_sampling_2d_region_3d_tf_wcons(rng: random.Generator, oracle_info_1: OracleCreator):

    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 3
    num_samples = 10

    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, oracle_info_1, rng)
 
def test_lhs_sampling_3d_region_2d_tf_wcons(rng: random.Generator, oracle_info_3: OracleCreator):

    region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
    tf_dim = 2
    num_samples = 10

    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, oracle_info_3, rng)

def test_lhs_sampling_region_desc_wcons(rng: random.Generator, oracle_info_1: OracleCreator):

    region_support = np.array([[1, -1], [1, -1]])
    tf_dim = 2
    num_samples = 10

    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, oracle_info_1, rng)
    
def test_lhs_sampling_array_shape_check_2dim_wcons(rng: random.Generator, oracle_info_1: OracleCreator, data_path: pathlib.Path):

    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 2
    num_samples = 10

    samples_in_unif = lhs_sampling(
        num_samples, region_support, tf_dim, oracle_info_1, rng
    )
    
    with open(data_path / "lhs_samp_wcons_t1.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)

    np.testing.assert_array_equal(samples_in_unif, gr)       

def test_lhs_sampling_array_shape_check_4dim_wcons(rng: random.Generator, oracle_info_4: OracleCreator, data_path: pathlib.Path):
    # oracle_info = OracleCreator(oracle_func_4d, 100,1)
    region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
    tf_dim = 4
    num_samples = 5

    samples_in_unif = lhs_sampling(
        num_samples, region_support, tf_dim, oracle_info_4, rng
    )
    
    with open(data_path / "lhs_samp_wcons_t2.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)

def test6_lhs_sampling(rng: random.Generator, oracle_info_4: OracleCreator):
    # oracle_info = OracleCreator(oracle_func_4d, 100,1)
    region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
    tf_dim = 4
    num_samples = 10

    with pytest.raises(OOBError):
        lhs_sampling(
            num_samples, region_support, tf_dim, oracle_info_4, rng
        )
