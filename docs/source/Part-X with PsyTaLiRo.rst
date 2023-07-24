.. part-x documentation master file, created by
   sphinx-quickstart on Wed Jan  5 07:23:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Part-X with PsyTaLiRo
===================


Black-Box Function
------------------
The black-box function is the function for whom we need to find the falsifying behaviors. Below, we show to two examples.

1) A blackbox function can also be defined which can be used for checking against a certain specification using `PSY-TaLiRo 1.0.0a14 <https://sbtg.gitlab.io/psy-taliro/>`_.
Here, we define the Automatic Transmission Blackbox.

.. code-block:: python

   ...

   class AutotransModel(Model[AutotransDataT, None]):
      MODEL_NAME = "Autotrans_shift"

      def __init__(self) -> None:
         if not _has_matlab:
               raise RuntimeError(
                  "Simulink support requires the MATLAB Engine for Python to be installed"
               )

         engine = matlab.engine.start_matlab()
         # engine.addpath("examples")
         model_opts = engine.simget(self.MODEL_NAME)

         self.sampling_step = 0.05
         self.engine = engine
         self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

      def simulate(self, static: StaticInput, signals: Signals, intrvl: Interval) -> AutotransResultT:
         sim_t = matlab.double([0, intrvl.upper])
         n_times = (intrvl.length // self.sampling_step) + 2
         signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
         signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in signals])

         model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())
         
         timestamps, _, data = self.engine.sim(
               self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
         )

         timestamps_array = np.array(timestamps).flatten()
         data_array = np.array(data)

         return ModelData(data_array.T, timestamps_array)

Oracle Function
------------------
The oracle function is the function which defines feasible and infeasible points that can be sampled by the Part-X algorithm. The oracle function takes in a point and returns value. The constraint is of the form :math:`f(x) <= 0`. 

Here is an example of simple constraint

.. code-block:: python
   
   # if constraint is X[0]**2 + (X[1]+1)**2 > 0.5, then
   def oracle_function(X):
      return -1*(X[0]**2 + (X[1]+1)**2-0.5)

Specification and Signals using Psy-TaLiRo:
-------------------------------------------

The Specification against which the signal are to be tested is written as follows:

.. code-block:: python

   AT1_phi = "G[0, 20] (speed <= 120)"
   specification = RTAMTDense(AT1_phi, {"speed": 0})

The Signal options are using psy-taliro. Below is an example:

In this example, we have two signals from time 0 to time 50. 

1. The first signal has 7 control points and the range for all the 7 control points are [0,100].

2. The second signal has 3 control points and the range for all the 3 control points are [0,325]:

.. code-block:: python

   signals = [
        SignalOptions(control_points = [(0, 100)]*7, signal_times=np.linspace(0.,50.,7)),
        SignalOptions(control_points = [(0, 325)]*3, signal_times=np.linspace(0.,50.,3)),
    ]

.. _gpr_definition_pxpsy:

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

.. _bo_definition_pxpsy:

Defining Bayesian Optimization Model
------------------------------------

Like the GPR, the Bayesian Optimization (BO) is also an essential part of the Part-X algorithm. 
While one can choose the internal BO model by importing it:

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


Parameters for Part-X
---------------------

The parameters that need to be defined for Part-X are mentioned below.
The usage of these parameters are defined in the :ref:`reference_examples`

- **BENCHMARK_NAME**: string
   Name of the benchmark

..

- **oracle_function**: function
   The oracle function

..

- **num_macro_rep**: int
   The number of replications

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

- **gpr_model**: The Gaussian Process Regressor model. Described in detail :ref:`_gpr_definition_pxpsy`.

..

- **bo_model**: The Bayesian Optimization model. Described in detail :ref:`_bo_definition_pxpsy`.

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

Defining the Options from psy-taliro
------------------------------------

The maximum budget and evaluation time for signal are defined using psy-taliro options.

.. code-block:: python

   options = Options(runs=1, iterations=self.MAX_BUDGET, interval=(0, 50),  signals=self.signals)



Running the Optimizer
----------------------

Once the all of it defined, we can run the code. 

.. code-block:: python

   ...

   staliro(model, specification, optimizer, options)

