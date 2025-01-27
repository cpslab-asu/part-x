import pickle
import numpy as np
import pytest
import pathlib
import time
from numpy import random

from partx.utilities.utils import branch_region, OracleCreator, calculate_volume, Fn, divide_points, compute_robustness
from partx.utilities.sampling import uniform_sampling, lhs_sampling

@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)

@pytest.fixture()
def oracle_info() -> OracleCreator:
 return OracleCreator(None, 1,1)

@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
    
def test_fn_initial_state():
    """
    Test the initial state of the Fn instance.
    """
    def dummy_func(x):
        return x * 2

    wrapped_func = Fn(dummy_func)
    assert wrapped_func.count == 0
    assert wrapped_func.point_history == []
    assert wrapped_func.simultation_time == []

def test_fn_single_call():
    """
    Test the behavior of the Fn instance for a single function call.
    """
    def dummy_func(x):
        return x * 2

    wrapped_func = Fn(dummy_func)
    result = wrapped_func(5)
    assert result == 10
    assert wrapped_func.count == 1
    assert len(wrapped_func.point_history) == 1
    assert wrapped_func.point_history[0] == [1, 5, 10]
    assert len(wrapped_func.simultation_time) == 1
    assert wrapped_func.simultation_time[0] >= 0  # Ensure the time is non-negative

def test_fn_multiple_calls():
    """
    Test the behavior of the Fn instance for multiple function calls.
    """
    def dummy_func(x, y):
        return x + y

    wrapped_func = Fn(dummy_func)
    inputs = [(1, 2), (3, 4), (5, 6)]
    expected_results = [3, 7, 11]

    for i, (args, expected) in enumerate(zip(inputs, expected_results), 1):
        result = wrapped_func(*args)
        assert result == expected
        assert wrapped_func.count == i
        assert wrapped_func.point_history[-1] == [i, *args, expected]
        assert len(wrapped_func.simultation_time) == i
        assert wrapped_func.simultation_time[-1] >= 0

def test_fn_timing():
    """
    Test that the timing information is recorded correctly.
    """
    def slow_func(x):
        time.sleep(0.1)  # Simulate a slow function
        return x ** 2

    wrapped_func = Fn(slow_func)
    result = wrapped_func(3)
    assert result == 9
    assert len(wrapped_func.simultation_time) == 1
    assert wrapped_func.simultation_time[0] >= 0.1  # Ensure the timing matches the sleep duration



def test_branch_region_split_br2_d0(rng: random.Generator):
    region_support = np.array([[-1., 1.], [-1., 2.]])
    direction_of_branching = 0
    uniform = True
    branching_factor = 2
    
    new_reg = branch_region(region_support, direction_of_branching, uniform ,branching_factor, rng)

    np.testing.assert_array_equal(new_reg[0], np.array([[-1., 0.], [-1., 2.]]))
    np.testing.assert_array_equal(new_reg[1], np.array([[0., 1.], [-1., 2.]]))

def test_branch_region_split_br4_d0(rng: random.Generator):
    region_support = np.array([[-1., 1.], [-1., 2.]])
    direction_of_branching = 0
    uniform = True
    branching_factor = 4
    
    new_reg = branch_region(region_support, direction_of_branching, uniform ,branching_factor, rng)
    
    np.testing.assert_array_equal(new_reg[0], np.array([[-1., -0.5], [-1., 2.]]))
    np.testing.assert_array_equal(new_reg[1], np.array([[-0.5, 0], [-1., 2.]]))
    np.testing.assert_array_equal(new_reg[2], np.array([[0, 0.5], [-1., 2.]]))
    np.testing.assert_array_equal(new_reg[3], np.array([[0.5, 1], [-1., 2.]]))

def test_branch_region_split_br2_d1(rng: random.Generator):
    region_support = np.array([[-1., 1.], [-1., 2.]])
    direction_of_branching = 1
    uniform = True
    branching_factor = 2
    
    new_reg = branch_region(region_support, direction_of_branching, uniform ,branching_factor, rng)
    
    np.testing.assert_array_equal(new_reg[0], np.array([[-1., 1.], [-1., 0.5]]))
    np.testing.assert_array_equal(new_reg[1], np.array([[-1., 1.], [0.5, 2.]]))

def test_branch_region_split_br2_b1(rng: random.Generator):
    region_support = np.array([[-1., 1.], [-1., 2.]])
    direction_of_branching = 1
    uniform = True
    branching_factor = 4
    
    new_reg = branch_region(region_support, direction_of_branching, uniform ,branching_factor, rng)
    
    np.testing.assert_array_equal(new_reg[0], np.array([[-1., 1.], [-1., -0.25]]))
    np.testing.assert_array_equal(new_reg[1], np.array([[-1., 1.], [-0.25, 0.5]]))
    np.testing.assert_array_equal(new_reg[2], np.array([[-1., 1.], [0.5, 1.25]]))
    np.testing.assert_array_equal(new_reg[3], np.array([[-1., 1.], [1.25, 2]]))

def test_calculate_volume():
    # Test 1: Simple 2D square region
    region_2d = np.array([[0, 2], [0, 3]])  # 2D rectangle with bounds
    assert calculate_volume(region_2d) == 6, "Test 1 Failed"
    
    # Test 2: Higher dimensions (3D hypercube)
    region_3d = np.array([[0, 2], [0, 3], [0, 4]])  # Bounds in 3D
    assert calculate_volume(region_3d) == 24, "Test 2 Failed"
    
    # Test 3: Single region (1D)
    region_1d = np.array([[0, 5]])  # 1D line
    assert calculate_volume(region_1d) == 5, "Test 3 Failed"
    
    # Test 4: Multiple regions (2D and 3D)
    regions_multiple = np.array([
        [[0, 2], [0, 2]],  # Region 1
        [[0, 3], [0, 3]],  # Region 2
    ])
    volumes = [calculate_volume(region) for region in regions_multiple]
    assert volumes == [4, 9], "Test 4 Failed"
    
    # Test 5: Edge case (zero width)
    region_zero_width = np.array([[1, 1], [2, 5]])  # Zero width in one dimension
    assert calculate_volume(region_zero_width) == 0, "Test 5 Failed"

def test1_divide_points(data_path: pathlib.Path, oracle_info: OracleCreator, rng: random.Generator):
    def test_function(X):
        return X[0] ** 2 + X[1] ** 2
    funct = Fn(test_function)
    
    region_support = np.array([[-1., 1.], [-1., 2.]])
    with open(data_path / "dataset_divideBranch.pickle", "rb") as f:
        samples = pickle.load(f)
        # pickle.dump(samples, f)
    
    samples_out = compute_robustness(samples, funct)

    
    reg_sup = branch_region(region_support, 1, False, 4, rng)
    x, y = divide_points(samples, samples_out, reg_sup)
    
    with open(data_path / "dataset_divideBranch_results.pickle", "rb") as f:
        gr_x, gr_y = pickle.load(f)
        # pickle.dump((x,y), f)

    for _x, _y, _gr_x, _gr_y in zip(x,y, gr_x, gr_y):
        np.testing.assert_array_equal(_x, _gr_x)
        np.testing.assert_array_equal(_y, _gr_y)

def test2_divide_points(rng: random.Generator):
    
    region_support = np.array([[-1., 1.], [-1., 2.]])
    
    samples = np.array([[]])
    samples_out = np.array([])

    reg_sup = branch_region(region_support, 1, False, 4, rng)
    x, y = divide_points(samples, samples_out, reg_sup)
    
    for _x, _y in zip(x,y):
        assert _x.shape == (1,0)
        assert _y.shape == (0,)

def test_divide_points_mismatched_shapes():
    """Test divide_points with mismatched input and output shapes."""
    samples_in = np.array([[1, 2], [3, 4], [5, 6]])
    samples_out = np.array([10, 20])  # Mismatched length
    region_support = [
        np.array([[0, 4], [1, 3]]),
        np.array([[5, 8], [5, 8]])
    ]

    regions, robustness = divide_points(samples_in, samples_out, region_support)

    # Expect empty arrays for regions and robustness
    for r in regions:
        assert r.size == 0

    for rb in robustness:
        assert rb.size == 0

def test_divide_points_empty_inputs():
    """Test divide_points with empty input arrays."""
    samples_in = np.array([]).reshape(0, 2)
    samples_out = np.array([])
    region_support = [
        np.array([[0, 4], [1, 3]]),
        np.array([[5, 8], [5, 8]])
    ]

    regions, robustness = divide_points(samples_in, samples_out, region_support)

    # Expect empty arrays for regions and robustness
    for r in regions:
        assert r.size == 0

    for rb in robustness:
        assert rb.size == 0

def test_divide_points_no_matching_regions():
    """Test divide_points where no samples match any region."""
    samples_in = np.array([[1, 2], [3, 4]])
    samples_out = np.array([10, 20])
    region_support = [
        np.array([[10, 20], [10, 20]])  # No matching regions
    ]

    regions, robustness = divide_points(samples_in, samples_out, region_support)

    assert len(regions) == 1
    assert len(robustness) == 1

    assert regions[0].size == 0
    assert robustness[0].size == 0

def test_divide_points_partial_matching_regions():
    """Test divide_points where only some samples match a region."""
    samples_in = np.array([[1, 2], [3, 4], [5, 6]])
    samples_out = np.array([10, 20, 30])
    region_support = [
        np.array([[0, 4], [0, 5]])  # Only matches the first two samples
    ]

    regions, robustness = divide_points(samples_in, samples_out, region_support)

    np.testing.assert_array_equal(regions[0], np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(robustness[0], np.array([10, 20]))

def test1_uniform_sampling(data_path:pathlib.Path):
    # Load gold resources for testing
    with open(data_path / "calRob_x_t1.pickle", "rb") as f:
        inputX = pickle.load(f)

    with open(data_path / "calRob_y_t1.pickle", "rb") as f:
        inputY = pickle.load(f)

    def test_function(X):
        return X[0] ** 2 + X[1] ** 2

    decFunction = Fn(test_function)
    
    # Run the robustness computation
    out = compute_robustness(inputX, decFunction)
    
    # Assert that the output matches the expected values
    np.testing.assert_array_equal(inputY, out)


def test_compute_robustness_single_sample():
    def test_function(X):
        return X[0] ** 2 + X[1] ** 2
    
    inputX = np.array([[1, 2]])  # Single sample
    decFunction = Fn(test_function)
    
    # Expected output for the test function (1^2 + 2^2 = 5)
    expected_output = np.array([5])
    
    out = compute_robustness(inputX, decFunction)
    
    np.testing.assert_array_equal(out, expected_output)


# Test 3: Multiple sample input
def test_compute_robustness_multiple_samples():
    def test_function(X):
        return X[0] ** 2 + X[1] ** 2
    
    inputX = np.array([[1, 2], [3, 4], [5, 6]])  # Multiple samples
    decFunction = Fn(test_function)
    
    # Expected output for each sample (1^2 + 2^2 = 5, 3^2 + 4^2 = 25, 5^2 + 6^2 = 61)
    expected_output = np.array([5, 25, 61])
    
    out = compute_robustness(inputX, decFunction)
    
    np.testing.assert_array_equal(out, expected_output)


# Test 4: Edge case with an empty input
def test_compute_robustness_empty_input():
    def test_function(X):
        return X[0] ** 2 + X[1] ** 2
    
    inputX = np.array([]).reshape(0, 2)  # Empty input (no samples)
    decFunction = Fn(test_function)
    
    # Expected output should also be empty
    expected_output = np.array([])
    
    with pytest.raises(ValueError):
        compute_robustness(inputX, decFunction)
    
    


# Test 5: Check if the function handles single-dimension arrays correctly
def test_compute_robustness_single_dimension():
    def test_function(X):
        return X[0] ** 2
    
    inputX = np.array([[1], [2], [3]])  # Single dimension input
    decFunction = Fn(test_function)
    
    # Expected output for each sample (1^2 = 1, 2^2 = 4, 3^2 = 9)
    expected_output = np.array([1, 4, 9])
    
    out = compute_robustness(inputX, decFunction)
    
    np.testing.assert_array_equal(out, expected_output)


# Test 6: Check function with a different test function
def test_compute_robustness_different_function():
    def test_function(X):
        return X[0] * X[1]  # Multiplication of the two inputs
    
    inputX = np.array([[2, 3], [4, 5], [6, 7]])  # Multiple samples
    decFunction = Fn(test_function)
    
    # Expected output for each sample (2*3 = 6, 4*5 = 20, 6*7 = 42)
    expected_output = np.array([6, 20, 42])
    
    out = compute_robustness(inputX, decFunction)
    
    np.testing.assert_array_equal(out, expected_output)