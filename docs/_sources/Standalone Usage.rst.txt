.. part-x documentation master file, created by
   sphinx-quickstart on Wed Jan  5 07:23:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Standalone Usage
=================


To run Part-X, we need a black-box function, and oracle function and to define various parameters of Part-X. 

Black-Box Function
------------------
The black-box function is the function for whom we need to find the falsifying behaviors. Below, we show to two examples.

 The non-linear, non-convex Himmelblaus Function can be defined as follows:

.. code-block:: python

   def test_function(X):
      return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40

.. _reference_oracle_function:
Oracle Function
------------------
The oracle function is the function which defines feasible and infeasible points that can be sampled by the Part-X algorithm. The oracle function takes in a point and returns value. The constraint is of the form :math:`f(x) <= 0`. 

Here is an example of simple constraint

.. code-block:: python
   
   # if constraint is X[0]**2 + (X[1]+1)**2 > 0.5, then
   def oracle_function(X):
      return -1*(X[0]**2 + (X[1]+1)**2-0.5)


If there are no constraints, you can define the oracle function as follows:

.. code-block:: python

   def oracle_function(X):
      return True

.. _reference_gpr_definition_standalone:
Defining GPR Model
-------------------

The Gaussian Process Regressor(GPR) is an essential part of the Part-X algorithm. 
While one can choose the internal GPR model by importing it:

.. code-block:: python

   from partx.gprInterface import InternalGPR
   gpr_model = InternalGPR()

This *gpr_model* can be passed to Part-X algorithm.

However, we even provide a way for users to use their own GPR. 

To use this, the user has to import the GPR interface and write their GPR model as shown below:

.. code-block:: python

   from .gprInterface import GaussianProcessRegressorStructure
   from sklearn.gaussian_process import GaussianProcessRegressor
   from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF, WhiteKernel
   from scipy.optimize import fmin_l_bfgs_b
   from sklearn.preprocessing import StandardScaler
   from warnings import catch_warnings
   import warnings


   def optimizer_lbfgs_b(obj_func, initial_theta):
      with catch_warnings():
         warnings.simplefilter("ignore")
         params = fmin_l_bfgs_b(
               obj_func, initial_theta, bounds=None, maxiter=30000, maxfun=1e10
         )
      return params[0], params[1]


   class UserDefinedGPR(GaussianProcessRegressorStructure):
      def __init__(self, random_state = 12345):
         self.gpr_model = GaussianProcessRegressor(
               kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state = random_state
         )
         self.scale = StandardScaler()

      def fit_gpr(self, X, Y):
         """Method to fit gpr Model

         Args:
               x_train: Samples from Training set.
               y_train: Evaluated values of samples from Trainig set.

         
         """
         X_scaled = self.scale.fit_transform(X)
         
         with catch_warnings():
               warnings.simplefilter("ignore")
               self.gpr_model.fit(X_scaled, Y)

      def predict_gpr(self, X):
         """Method to predict mean and std_dev from gpr model

         Args:
               x_train: Samples from Training set.
               

         Returns:
               mean
               std_dev
         """
         x_scaled = self.scale.transform(X)
         with catch_warnings():
               warnings.simplefilter("ignore")
               yPred, predSigma = self.gpr_model.predict(x_scaled, return_std=True)
         return yPred, predSigma

.. _reference_bo_definition_standalone:
Defining Bayesian Optimization Model
------------------------------------

Like the GPR, the Bayesian Optimization (BO) is also an essential part of the Part-X algorithm. 
While one can choose the internal BO model by importing it:

.. code-block:: python

   from partx.bayesianOptimization import InternalBO
   bo_model = InternalBO()

This *bo_model* can be passed to Part-X algorithm.

However, we even provide a way for users to use their own BO code. The idea is that the user can plug in the existing BO implementation for an implementatin such that it returns a single new point.

To use this, the user has to import the BO interface and write their BO model as shown below:

.. code-block:: python

  from typing import Callable, Tuple
   import numpy as np
   from numpy.typing import NDArray
   from scipy.optimize import minimize
   from scipy.stats import norm

   from .bointerface import BO_Interface
   from ..gprInterface import GPR
   from ..sampling import uniform_sampling

   class InternalBO(BO_Interface):
      def __init__(self):
         pass

      def sample(
         self,
         x_train: NDArray,
         y_train: NDArray,
         region_support: NDArray,
         gpr_model: Callable,
         oracle_info,
         rng,
      ) -> Tuple[NDArray]:

         """Internal BO Model

         Args:
               x_train: Samples from Training set.
               y_train: Evaluated values of samples from Trainig set.
               region_support: Min and Max of all dimensions
               gpr_model: Gaussian Process Regressor Model developed using Factory
               oracle_info: Oracle defining the constraints.
               rng: RNG object from numpy

         Raises:
               TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
               TypeError: If y_train is not (n,) numpy array
               TypeError: If there is a mismatch between x_train and y_train

         Returns:
               x_new
         """
         
         model = GPR(gpr_model)
         model.fit(x_train, y_train)

         pred_sample_x = self._opt_acquisition(y_train, model, region_support, oracle_info, rng)


         return pred_sample_x

      def _opt_acquisition(self, y_train: NDArray, gpr_model: Callable, region_support: NDArray, oracle_info, rng) -> NDArray:
         """Get the sample points

         Args:
               X: sample points
               y: corresponding robustness values
               model: the GP models
               sbo: sample points to construct the robustness values
               test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)
               region_support: The bounds of the region within which the sampling is to be done.
                                          Region Bounds is M x N x O where;
                                             M = number of regions;
                                             N = test_function_dimension (Dimensionality of the test function);
                                             O = Lower and Upper bound. Should be of length 2;

         Returns:
               The new sample points by BO
         """

         tf_dim = region_support.shape[0]
         lower_bound_theta = np.ndarray.flatten(region_support[:, 0])
         upper_bound_theta = np.ndarray.flatten(region_support[:, 1])

         curr_best = np.min(y_train)

         # bnds = Bounds(lower_bound_theta, upper_bound_theta)
         fun = lambda x_: -1 * self._acquisition(y_train, x_, gpr_model)

         random_samples = uniform_sampling(2000, region_support, tf_dim, oracle_info, rng)
         min_bo_val = -1 * self._acquisition(
               y_train, random_samples, gpr_model, "multiple"
         )

         min_bo = np.array(random_samples[np.argmin(min_bo_val), :])
         min_bo_val = np.min(min_bo_val)

         for _ in range(9):
               new_params = minimize(
                  fun,
                  bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                  x0=min_bo,
               )

               if not new_params.success:
                  continue

               if min_bo is None or fun(new_params.x) < min_bo_val:
                  min_bo = new_params.x
                  min_bo_val = fun(min_bo)
         new_params = minimize(
               fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
         )
         min_bo = new_params.x

         return np.array(min_bo)

      def _surrogate(self, gpr_model: Callable, x_train: NDArray):
         """_surrogate Model function

         Args:
               model: Gaussian process model
               X: Input points

         Returns:
               Predicted values of points using gaussian process model
         """

         return gpr_model.predict(x_train)

      def _acquisition(self, y_train: NDArray, sample: NDArray, gpr_model: Callable, sample_type:str ="single") -> NDArray:
         """Acquisition Model: Expected Improvement

         Args:
               y_train: corresponding robustness values
               sample: Sample(s) whose EI is to be calculated
               gpr_model: GPR model
               sample_type: Single sample or list of model. Defaults to "single". other options is "multiple".

         Returns:
               EI of samples
         """
         curr_best = np.min(y_train)

         if sample_type == "multiple":
               mu, std = self._surrogate(gpr_model, sample)
               ei_list = []
               for mu_iter, std_iter in zip(mu, std):
                  pred_var = std_iter
                  if pred_var > 0:
                     var_1 = curr_best - mu_iter
                     var_2 = var_1 / pred_var

                     ei = (var_1 * norm.cdf(var_2)) + (
                           pred_var * norm.pdf(var_2)
                     )
                  else:
                     ei = 0.0

                  ei_list.append(ei)
               # print(np.array(ei_list).shape)
               # print("*****")
               # return np.array(ei_list)
         elif sample_type == "single":
               # print("kfkf")
               mu, std = self._surrogate(gpr_model, sample.reshape(1, -1))
               pred_var = std[0]
               if pred_var > 0:
                  var_1 = curr_best - mu[0]
                  var_2 = var_1 / pred_var

                  ei = (var_1 * norm.cdf(var_2)) + (
                     pred_var * norm.pdf(var_2)
                  )
               else:
                  ei = 0.0
               # return ei

         if sample_type == "multiple":
               return_ei = np.array(ei_list)
         elif sample_type == "single":
               return_ei = ei

         return return_ei



Parameters
----------

The parameters that need to be defined for Part-X are mentioned below.
The usage of these parameters are defined in the :ref:`reference_examples`

- **BENCHMARK_NAME**: string
   Name of the benchmark

..

- **test_function**: function
   The black-box test function

..

- **oracle_function**: function
   The oracle function

..

- **num_macro_rep**: int
   The number of replications

..

- **init_reg_sup**: 2-d Numpy Array
   Needs to be a 3-dimensional list that represents the initial region support of the function.
   For exmples: 

   .. code-block:: python

      region_support = [[-5,5], [-2,3], [-3,4]]
   
   Here, the first dimension has the range [-5,5], the second dimension has the range [-2,3] and theird dimension has the range [-3,4]

..

- **tf_dim**: int
   Needs to be an interger that represents the dimensionality of the blask-box function

..
- **max_budget**: int
   The maximum budget or the maximum number of evaluations of the black-box function that are allowed.

..
- **init_budget**: int
   The initiliazation budget of the algorithm. This refers to minimimum nunmber of samples that are required to be present in a region in order to generate samples from bayesian optimization and classify the region.

..

- **bo_budget**: int
   The number of samples that needs to be generated from Bayesian Optimization

..

- **cs_budget**: int
   The number of samples that must sampled from continuous sampling phase.

..

- **n_tries_randomsampling**: int
   The number of tries a point should be sampled again to follow constraints in the random sampling phase. In case of no no constraint, set the value to 1. An error is raised if the number of tries is exhausted.

..

- **n_tries_BO**: int
   The number of tries a point should be sampled again to follow constraints in the BO sampling phase. In case of no constraint, set the value to 1. If th number of tries is exhausted, a random feasible point is selected.

..

- **alpha**: float, [0,1]
   Region Classification percentile

..

- **R**: int
   The number of monte-carlo iterations. This is used in calculation of quantiles of a region.

..

- **M**: int
   The number of evaluation of per monte-carlo iteration. This is used in calculation of quantiles of a region.

..


- **delta**: float, int
   A number used to define the fraction of dimension, below which no further brnching in that dimension takes place. It is used for clsssificastion of a region.

..

- **fv_quantiles_for_gp** list
   List of values used for calculation at certain quantile values.

..



- **branching_factor**: int
   Number of sub-regions in which a region is branched. 

..

- **uniform_partitioning** True/False
   Wether to perform Uniform Partitioning or not. 

.. 

- **start_seed**: int
   Starting seed of the experiment to ensure reproducibility.

..

- **gpr_model**: The Gaussian Process Regressor model. Described in detail :ref:`_reference_gpr_definition_standalone` .

..

- **bo_model**: The Bayesian Optimization model. Described in detail :ref:`_reference_bo_definition_standalone` .

..

- **init_sampling_type**: str
   Initial Sampling Algorithms. Defaults to "lhs_sampling". Can also use "uniform_sampling"
..

- **cs_sampling_type**: str
   Continued Sampling Mechanism. Defaults to "lhs_sampling". Can also use "uniform_sampling"
..

- **q_estim_sampling**: str
   Quantile estimation sampling Mechanism. Defaults to "lhs_sampling". Can also use "uniform_sampling"
..

- **mc_integral_sampling_type**: str
   Monte Carlo Integral Sampling Mechanism. Defaults to "lhs_sampling". Can also use "uniform_sampling"
..

- **results_sampling_type**: str
   Results Sampling Mechanism. Defaults to "lhs_sampling". Can also use "uniform_sampling"
..

- **results_at_confidence**: float
   Confidence level at which result to be computed
..

- **results_folder_name**: 
   Results folder name, 
..

- **num_cores**: int
   Number of cores to use. If value is 1, no parallalization is used. If value is greater than 1, various macro-replication will be spread over the cores.

It would be advisable to refer to Algorithm 1, 2, 3, 4 in the paper `Part-X <https://arxiv.org/pdf/2110.10729.pdf>`_ to get a deeper understanding of these paramters and where they are used.


Running the Optimizer
----------------------

Once the black-box function and the parameters are defined, we can run the code. 

If we are using psy-staliro and passing the Part-X as an optimizer, we csn define the parameters as follows and pass them as options to psy-staliro. 

.. code-block:: python

   ...

   from partx.partxInterface import run_partx

   run_partx(BENCHMARK_NAME, 
            test_function, 
            oracle_function,
            num_macro_reps, 
            init_reg_sup, 
            tf_dim,
            max_budget, 
            init_budget, 
            bo_budget, 
            cs_budget, 
            n_tries_randomsampling,
            n_tries_BO,
            alpha, 
            R, 
            M, 
            delta, 
            fv_quantiles_for_gp,
            branching_factor, 
            uniform_partitioning, 
            start_seed,
            gpr_model, 
            bo_model, 
            init_sampling_type, 
            cs_sampling_type, 
            q_estim_sampling, 
            mc_integral_sampling_type, 
            results_sampling_type, 
            results_at_confidence, 
            results_folder_name, 
            num_cores) 

..

   