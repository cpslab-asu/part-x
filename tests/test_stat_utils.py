import pickle
import numpy as np
import pytest
import pathlib
from numpy import random

from partx.utilities.utils import OracleCreator, Fn, compute_robustness
from partx.utilities.sampling import lhs_sampling, uniform_sampling
from partx.utilities.stat_utils import calculate_mc_integral, conf_interval, assign_budgets, calculate_quantile, estimate_quantiles, mc_step
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

def test_simple_case():
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

def test_simple_case_assignbudgets():
    volumes = [0.25, 0.25, 0.25, 0.25]
    cs_budget = 1000
    result = assign_budgets(volumes, cs_budget)
    assert np.sum(result) == cs_budget
    assert all([abs(250 - r) < 40 for r in result])  # Allow some tolerance due to randomness

def test_uneven_volumes():
    volumes = [0.1, 0.3, 0.6]
    cs_budget = 500
    result = assign_budgets(volumes, cs_budget)
    assert np.sum(result) == cs_budget
    expected_distribution = [50, 150, 300]
    assert all(abs(r - e) < 50 for r, e in zip(result, expected_distribution))

def test_single_volume():
    volumes = [1.0]
    cs_budget = 100
    result = assign_budgets(volumes, cs_budget)
    assert result == [100]

def test_zero_budget():
    volumes = [0.5, 0.5]
    cs_budget = 0
    result = assign_budgets(volumes, cs_budget)
    assert result == [0, 0]

def test_no_volumes():
    volumes = []
    cs_budget = 100
    result = assign_budgets(volumes, cs_budget)
    assert result == []

def test_large_budget():
    volumes = [0.2, 0.3, 0.5]
    cs_budget = 1_000_000
    result = assign_budgets(volumes, cs_budget)
    assert sum(result) == cs_budget
    expected_distribution = [200_000, 300_000, 500_000]
    assert all(abs(r - e) < 5000 for r, e in zip(result, expected_distribution))  # Larger tolerance for large budgets

def test_all_zero_volumes():
    volumes = [0, 0, 0]
    cs_budget = 100
    result = assign_budgets(volumes, cs_budget)
    assert result == [0, 0, 0]


def test_calculate_quantile_valid_input():
    y_pred = np.array([1.0, 2.0, 3.0])
    sigma_st = np.array([0.1, 0.2, 0.3])
    alpha = 0.05

    lower_quantile, upper_quantile = calculate_quantile(y_pred, sigma_st, alpha)

    expected_term2 = stats.norm.ppf(1 - (alpha / 2)) * sigma_st

    assert np.allclose(lower_quantile, y_pred + expected_term2), "Lower quantile calculation is incorrect"
    assert np.allclose(upper_quantile, y_pred - expected_term2), "Upper quantile calculation is incorrect"

def test_calculate_quantile_zero_sigma():
    y_pred = np.array([1.0, 2.0, 3.0])
    sigma_st = np.array([0.0, 0.0, 0.0])
    alpha = 0.05

    lower_quantile, upper_quantile = calculate_quantile(y_pred, sigma_st, alpha)

    assert np.allclose(lower_quantile, y_pred), "Lower quantile should equal y_pred when sigma_st is zero"
    assert np.allclose(upper_quantile, y_pred), "Upper quantile should equal y_pred when sigma_st is zero"

def test_calculate_quantile_large_alpha():
    y_pred = np.array([1.0, 2.0, 3.0])
    sigma_st = np.array([0.1, 0.2, 0.3])
    alpha = 0.99

    lower_quantile, upper_quantile = calculate_quantile(y_pred, sigma_st, alpha)

    expected_term2 = stats.norm.ppf(1 - (alpha / 2)) * sigma_st

    assert np.allclose(lower_quantile, y_pred + expected_term2), "Lower quantile calculation is incorrect for large alpha"
    assert np.allclose(upper_quantile, y_pred - expected_term2), "Upper quantile calculation is incorrect for large alpha"

def test_calculate_quantile_empty_arrays():
    y_pred = np.array([])
    sigma_st = np.array([])
    alpha = 0.05

    lower_quantile, upper_quantile = calculate_quantile(y_pred, sigma_st, alpha)

    assert lower_quantile.size == 0, "Lower quantile should be empty for empty input arrays"
    assert upper_quantile.size == 0, "Upper quantile should be empty for empty input arrays"

def test_mc_step_1(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    def internal_function(X):
        return (1 + (X[0] + X[1] + 1) ** 2 * (
            19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                    30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                        18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

    tf = Fn(internal_function)
    region_support = np.array([[-1., 1.], [-1., 1.]])
    tf_dim = 2

    R = 20
    M = 500
    gpr_model = InternalGPR()
    x_train = uniform_sampling(200, region_support, tf_dim, oracle_info, rng)
    y_train = compute_robustness(x_train, tf)
    alpha = 0.05

    
    
    
    min_quantile, max_quantile = mc_step(x_train, y_train, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type = "lhs_sampling")
    
    with open(data_path / "mc_step_1_minquantile.pickle", "rb") as f:
        # pickle.dump((min_quantile, max_quantile),f)
        min_q_gr, max_q_gr = pickle.load(f)
    
    np.testing.assert_almost_equal(min_quantile, min_q_gr, 1)
    np.testing.assert_almost_equal(max_quantile, max_q_gr, 1)

def test2_mc_step(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    def internal_function(X):
        return (1 + (X[0] + X[1] + 1) ** 2 * (
            19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                    30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                        18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

    tf = Fn(internal_function)
    region_support = np.array([[-1., 1.], [-1., 1.]])
    tf_dim = 2

    R = 20
    M = 500
    gpr_model = InternalGPR()
    x_train = uniform_sampling(100, region_support, tf_dim, oracle_info, rng)
    y_train = compute_robustness(x_train, tf)
    alpha = 0.05

    min_quantile, max_quantile = mc_step(x_train, y_train, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type = "uniform_sampling")

    with open(data_path / "mc_step_1_ufs_minquantile.pickle", "rb") as f:
        # pickle.dump((min_quantile, max_quantile),f)
        min_q_gr, max_q_gr = pickle.load(f)
    
    np.testing.assert_almost_equal(min_quantile, min_q_gr, 1)
    np.testing.assert_almost_equal(max_quantile, max_q_gr, 1)

def test3_estimateMC(rng: random.Generator, oracle_info: OracleCreator, data_path: pathlib.Path):
    region_support = np.array([[-1., 1.], [-1., 1.]])
    tf_dim = 2

    R = 100
    M = 10000
    gpr_model = InternalGPR()
    alpha = 0.05

    with open(data_path / "mc_quantile.pickle", "rb")  as f:
        gr_x_train, gr_y_train_std, gr_min_delta_quantile, gr_max_delta_quantile = pickle.load(f)

    min_delta_quantile, max_delta_quantile = estimate_quantiles(gr_x_train, gr_y_train_std, region_support, tf_dim, alpha, R, M, gpr_model, oracle_info, rng, sampling_type = "lhs_sampling")

    np.testing.assert_almost_equal(min_delta_quantile, gr_min_delta_quantile, decimal = 2)
    np.testing.assert_almost_equal(max_delta_quantile, gr_max_delta_quantile, decimal = 2)
