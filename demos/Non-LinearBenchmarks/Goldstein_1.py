from partx import r
import numpy as np
from partx.bayesianOptimization import InternalBO
from partx.gprInterface import InternalGPR

# import src.partx
def test_function(X):
    return (1 + (X[0] + X[1] + 1) ** 2 * (
                19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                       30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                           18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

BENCHMARK_NAME = "Goldstein_1"
init_reg_sup = np.array([[-1., 1.], [-1., 1.]])
tf_dim = 2
max_budget = 500
init_budget = 20
bo_budget = 20
cs_budget = 100
alpha = 0.05
R = 10
M = 100
delta = 0.001
fv_quantiles_for_gp = [0.5,0.05,0.01]
branching_factor = 2
uniform_partitioning = True
start_seed = 12345
gpr_model = InternalGPR()
bo_model = InternalBO()

init_sampling_type = "lhs_sampling"
cs_sampling_type = "lhs_sampling"
q_estim_sampling = "lhs_sampling"
mc_integral_sampling_type = "lhs_sampling"
results_sampling_type = "lhs_sampling"
results_at_confidence = 0.95


num_macro_reps = 5
results_folder_name = "NLF_1"
num_cores = 1
x = run_partx(BENCHMARK_NAME, test_function, num_macro_reps, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, 
                init_sampling_type, cs_sampling_type, 
                q_estim_sampling, mc_integral_sampling_type, 
                results_sampling_type, 
                results_at_confidence, results_folder_name, num_cores) 