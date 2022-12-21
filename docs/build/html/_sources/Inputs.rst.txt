.. part-x documentation master file, created by
   sphinx-quickstart on Wed Jan  5 07:23:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Inputs
=======


To run Part-X, we need a black-box function and initialize the parameters of Part-X. 

Black-Box Function
------------------
The black-box function is the function for whom we need to find the falsifying behaviors. Below, we show to two examples.

1) The non-linear, non-convex Himmelblaus Function can be defined as follows:

.. code-block:: python

   def test_function(X):
      return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40


2) A blackbox function can also be defined which can be used for checking against a certain specification using `PSY-TaLiRo 1.0.0a14 <https://sbtg.gitlab.io/psy-taliro/>`_.
Here, we define the F16 Auto GCAS black-box function.

.. code-block:: python

   @blackbox()
   def f16_model(static: StaticInput, times: SignalTimes, signals: SignalValues) -> F16DataT:
      power = 9
      alpha = np.deg2rad(2.1215)
      beta = 0
      alt = ALTITUDE
      vel = 540
      phi = static[0]
      theta = static[1]
      psi = static[2]

      initial_state = [vel, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
      step = 1 / 30
      autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str = "old")

      result = run_f16_sim(initial_state, max(times), autopilot, step, extended_states=True)
      trajectories: NDArray[np.float_] = result["states"][:, 11:12]
      timestamps: NDArray[np.float_] = result["times"]
      return ModelData(trajectories.T, timestamps)

Parameters
----------

The parameters that need to be defined for Part-X are mentioned below.
The usage of these parameters are defined in the :ref:`reference_examples`

- **BENCHMARK_NAME**: string
   Name of the benchmark

..

- **test_function**: function
   The black box test function

..

- **test_function_dimension**: int
   Needs to be an interger that represents the dimensionality of the blask-box function

..

- **region_support**: list of list of list
   Needs to be a 3-dimensional list that represents the initial region support of the function.
   For exmples: 

   .. code-block:: python

      region_support = [[[-5,5], [-2,3], [-3,4]]]
   
   Here, the first dimension has the range [-5,5], the second dimension has the range [-2,3] and theird dimension has the range [-3,4]

..

- **initialization_budget**: int
   The initiliazation budget of the algorithm. This refers to minimimum nunmber of samples that are required to be present in a region in order to generate samples from bayesian optimization and classify the region.

..

- **max_budget**: int
   The maximum budget or the maximum number of evaluations of the black-box function that are allowed.

..

- **continued_sampling_budget**: int
   The number of samples that must sampled from continuous sampling phase.

..

- **number_of_BO_samples**: int
   The number of samples that needs to be generated from Bayesian Optimization

..

- **M**: int
   The number of evaluation of per monte-carlo iteration. This is used in calculation of quantiles of a region.

..

- **R**: int
   The number of monte-carlo iterations. This is used in calculation of quantiles of a region.

..

- **branching_factor**: int
   Number of sub-regions in which a region is branched. 

..

- **alpha**: float, [0,1]
   Region Classification percentile

..

- **delta**: float, int
   A number used to define the fraction of dimension, below which no further brnching in that dimension takes place. It is used for clsssificastion of a region.

..

- **number_of_macro_replications**: int
   The number of replications

..

- **start_seed**: int
   Starting seed of the experiment to ensure reproducibility.

..

- **fv_quantiles_for_gp** list
   List of values used for calculation at certain quantile values.

..

- **results_at_confidence**: float
   Used to calculate the falsification volume at certain results_at_confidence

..

- **gpr_params**: list
   As of now, we have ready support for the following:
      1) An inbuilt library that utilises the Kriging Model with nugget effect.
         This library takes in an hyperparameter (`kriging_parameter`) as an input which decides the accuracy of the fit, and thus needs to be passed. 
         To run this inbuilt krigin model, `gpr_params` can be defined in the following way: 

         .. code-block:: python

            gpr_params = list(["kriging", kriging_parameter])
      
      2) Sklearn Gaussian Process Regressors. More information can be found here at `Scikit Learn Package <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_
         In order to run models from this library, first define the model and then pass it to gpr params. 

         Following is an example where we run the gaussain process regressor with Matern Kernel (nu = 1.5).

         .. code-block:: python

            gpr_model = GaussianProcessRegressor(
                                 kernel=Matern(nu=1.5),
                                 normalize_y=True,
                                 alpha=1e-6,
                                 n_restarts_optimizer=5)
            
            gpr_params = list(["other", gpr_model])

..

- **results_folder_name**: string
   Name of the results folder where the intermediate generated files will be stored.

..

- **num_cores**: int
   Number of cores to use. If value is 1, no parallalization is used. If value is greater than 1, various macro-replication will be spread over the cores.

It would be advisable to refer to Algorithm 1, 2, 3, 4 in the paper `Part-X <https://arxiv.org/pdf/2110.10729.pdf>`_ to get a deeper understanding of these paramtersa nd where they are used.


Running the Optimizer
----------------------

Once the black-box function and the parameters are defined, we can run the code. 

If we are using psy-staliro and passing the Part-X as an optimizer, we csn define the parameters as follows and pass them as options to psy-staliro. 

.. code-block:: python

   ...

   optimizer = PartX(
         benchmark_name="f16_alt{}_budget_{}".format(str(ALTITUDE).replace(".","_"), MAX_BUDGET),
         test_function_dimension = 3,
         initialization_budget = 30,
         continued_sampling_budget=100,
         number_of_BO_samples=[10],
         M = 500,
         R = 20,
         branching_factor=2,
         alpha=[0.05],
         delta=0.001,
         macro_replication=NUMBER_OF_MACRO_REPLICATIONS,
         fv_quantiles_for_gp = [0.5,0.05, 0.01],
         results_at_confidence = 0.95,
         gpr_params = 8,
         results_folder_name = "results",
         num_cores = 2
      )

   options = Options(runs=1, iterations=MAX_BUDGET, interval=(0, 15), static_parameters=initial_conditions, signals = [])

   result = staliro(
               f16_model,
               specification,
               optimizer,
               options
         )
