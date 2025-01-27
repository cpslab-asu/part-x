import pickle
import numpy as np
import pytest
import pathlib
from numpy import random

from partx.utilities.utils import OracleCreator
from partx.utilities.stat_utils import calculate_mc_integral, conf_interval
from partx.gpr import InternalGPR
from scipy import stats

@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)

@pytest.fixture()
def oracle_info() -> OracleCreator:
 return OracleCreator(None, 1,1)

@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"

def test_basic_case():
    x = np.array([10, 20, 30, 40, 50])
    conf_at = 0.95
    expected = stats.norm.interval(conf_at, loc=x.mean(), scale=x.std(ddof=1))
    assert np.allclose(conf_interval(x, conf_at), expected), "Basic test case failed"

def test_single_element():
    x = np.array([42])
    conf_at = 0.95
    with pytest.raises(ValueError):
        conf_interval(x, conf_at)  # Single-element datasets should raise an error

def test_empty_dataset():
    x = np.array([])
    conf_at = 0.95
    with pytest.raises(ValueError):
        conf_interval(x, conf_at)  # Empty datasets should raise an error

def test_high_confidence_level():
    x = np.array([5, 10, 15, 20, 25])
    conf_at = 0.99
    expected = stats.norm.interval(conf_at, loc=x.mean(), scale=x.std(ddof=1))
    assert np.allclose(conf_interval(x, conf_at), expected), "High confidence level test failed"

def test_low_confidence_level():
    x = np.array([5, 10, 15, 20, 25])
    conf_at = 0.80
    expected = stats.norm.interval(conf_at, loc=x.mean(), scale=x.std(ddof=1))
    assert np.allclose(conf_interval(x, conf_at), expected), "Low confidence level test failed"

def test_negative_values():
    x = np.array([-10, -20, -30, -40, -50])
    conf_at = 0.95
    expected = stats.norm.interval(conf_at, loc=x.mean(), scale=x.std(ddof=1))
    assert np.allclose(conf_interval(x, conf_at), expected), "Negative values test failed"

def test_custom_confidence_level():
    x = np.array([1, 2, 3, 4, 5])
    conf_at = 0.50
    expected = stats.norm.interval(conf_at, loc=x.mean(), scale=x.std(ddof=1))
    assert np.allclose(conf_interval(x, conf_at), expected), "Custom confidence level test failed"

def test_large_dataset():
    x = np.random.normal(100, 15, 1000)
    conf_at = 0.95
    expected = stats.norm.interval(conf_at, loc=x.mean(), scale=x.std(ddof=1))
    assert np.allclose(conf_interval(x, conf_at), expected), "Large dataset test failed"

def test_high_variability():
    x = np.array([1, 100, 200, 300, 400])
    conf_at = 0.95
    expected = stats.norm.interval(conf_at, loc=x.mean(), scale=x.std(ddof=1))
    assert np.allclose(conf_interval(x, conf_at), expected), "High variability test failed"

def test_low_variability():
    x = np.array([50, 51, 52, 53, 54])
    conf_at = 0.95
    expected = stats.norm.interval(conf_at, loc=x.mean(), scale=x.std(ddof=1))
    assert np.allclose(conf_interval(x, conf_at), expected), "Low variability test failed"

def test_invalid_confidence_level():
    x = np.array([10, 20, 30, 40, 50])
    with pytest.raises(ValueError):
        conf_interval(x, -0.1)  # Invalid confidence level

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