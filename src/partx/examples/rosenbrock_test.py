from ..interfaces.run_standalone import run_partx


def test_function(X):
    return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) - 20 # Rosenbrock


results = run_partx(
    benchmark_name="rosenbrock_test",
    test_function=test_function,
    test_function_dimension=2,
    region_support=np.array([[[-1., 1.], [-1., 1.]]]),
    initialization_budget=10,
    maximum_budget=5000,
    continued_sampling_budget=100,
    number_of_BO_samples=[10],
    number_of_samples_gen_GP=100,
    branching_factor=2,
    nugget_mean=0,
    nugget_std_dev=0.001,
    alpha=[0.95],
    delta=0.001,
    number_of_macro_replications=50,
    initial_seed=1000
)
