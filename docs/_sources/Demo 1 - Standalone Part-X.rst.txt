
Demo 1 - Standalone Part-X
===========================


Example - Running Part-X on Goldstein Price Function:
------------------------------------------------------

.. code-block:: python 

   from partx.partxInterface import run_partx
      import numpy as np
      from partx.bayesianOptimization import InternalBO
      from partx.gprInterface import InternalGPR


      # Define the Goldstein Price Test Function
      def test_function(X):
      return (1 + (X[0] + X[1] + 1) ** 2 * (
                  19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
                        30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                              18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

      # Define the Oracle Function which defines the constraints.
      # Since there is no constraint, return True
      oracle_func = None

      # Define Benchmark Name
      BENCHMARK_NAME = "Goldstein_1"

      # Define the Initial Search space. Here, we set it to [-1,1] on both the dimensions
      init_reg_sup = np.array([[-1., 1.], [-1., 1.]])

      # Function Dimesnioanlity set to 2 since we are searching in the 2-dimensional space
      tf_dim = 2

      # Max Budget is set to 500
      max_budget = 500

      # Initial Sampling in the subregion is set to 20
      init_budget = 20

      # BO sampling in each subregion is set to 20
      bo_budget = 20

      # Continued Sampling for subregions is set to 100
      cs_budget = 100

      # Define n_tries. Since there are no constraints involved, set them to 1
      n_tries_random_sampling = 1
      n_tries_BO = 1

      # Alpha, for Region Calssification percentile is set to 0.05 
      alpha = 0.05

      # R and M for quantile estimation in subregions is set 10 and 100 respectively
      R = 10
      M = 100

      # Minimum subregion cutoff is set 0.001. Anything less than 0.001 of the voulme of the hypercube will be calssified as unknown
      delta = 0.001

      # Helps in Result Calculation. Here, we want to obtain results at 50%, 95% and 99%.
      fv_quantiles_for_gp = [0.5,0.05,0.01]

      # Every time a subregion is branched, it branches into 2 non-overallping regions
      branching_factor = 2

      # If true, perform branching such that region is divided into two subregions
      uniform_partitioning = True

      # Starting seed
      start_seed = 12345

      # Using Internal GPR and BO model
      gpr_model = InternalGPR()
      bo_model = InternalBO()

      # Defining the sampling types
      init_sampling_type = "lhs_sampling"
      cs_sampling_type = "lhs_sampling"
      q_estim_sampling = "lhs_sampling"
      mc_integral_sampling_type = "lhs_sampling"
      results_sampling_type = "lhs_sampling"
      results_at_confidence = 0.95

      # Run Part-X for 5 macro-replications
      num_macro_reps = 5

      # All benchmarks will be stored in this folder
      results_folder_name = "NLF"

      # Run all the replication serially. If > 1, will run the replications parallaly.
      num_cores = 1

      # Run Part-X
      results = run_partx(BENCHMARK_NAME, test_function, oracle_fn, num_macro_reps, init_reg_sup, tf_dim,
                  max_budget, init_budget, bo_budget, cs_budget, n_tries_random_sampling, n_tries_BO,
                  alpha, R, M, delta, fv_quantiles_for_gp,
                  branching_factor, uniform_partitioning, start_seed, 
                  gpr_model, bo_model, 
                  init_sampling_type, cs_sampling_type, 
                  q_estim_sampling, mc_integral_sampling_type, 
                  results_sampling_type, 
                  results_at_confidence, results_folder_name, num_cores) 