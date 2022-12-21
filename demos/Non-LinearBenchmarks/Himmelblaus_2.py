from partx.partxInterface import run_partx
import numpy as np
from partx.bayesianOptimization import InternalBO
from partx.gprInterface import InternalGPR


def test_function(X):
    return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 


BENCHMARK_NAME = "Himmelblaus_1"
init_reg_sup = np.array([[-5., 5.], [-5., 5.]])
tf_dim = 2
max_budget = 5000
init_budget = 20
bo_budget = 20
cs_budget = 100
alpha = 0.05
R = 10
M = 500
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


num_macro_reps = 50
results_folder_name = "NLF"
num_cores = 10
x = run_partx(BENCHMARK_NAME, test_function, num_macro_reps, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, 
                init_sampling_type, cs_sampling_type, 
                q_estim_sampling, mc_integral_sampling_type, 
                results_sampling_type, 
                results_at_confidence, results_folder_name, num_cores) 