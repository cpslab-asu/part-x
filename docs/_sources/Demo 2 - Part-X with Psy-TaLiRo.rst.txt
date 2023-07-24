
Demo 2 - Part-X with Psy-TaLiRo
===============================


Example - Running Part-X on AT1 Specification:
------------------------------------------------------

We define the model as follows:

.. code-block:: python

   import numpy as np
   from numpy.typing import NDArray
   from staliro.core.interval import Interval
   from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
   from staliro.options import Options, SignalOptions
   from staliro.specifications import RTAMTDense
   from staliro.staliro import staliro, simulate_model


   try:
      import matlab
      import matlab.engine
   except ImportError:
      _has_matlab = False
   else:
      _has_matlab = True

   AutotransDataT = NDArray[np.float_]
   AutotransResultT = ModelData[AutotransDataT, None]


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


We then run part-X as follows:

.. code-block:: python 

   # Import All the necessary packges
   
   from AT_benchmark.AT_specifications import load_specification_dict
   from models import AutotransModel
   from Benchmark import Benchmark
   from partx.partxInterface.staliroIntegration import PartX
   from partx.bayesianOptimization.internalBO import InternalBO
   from partx.gprInterface.internalGPR import InternalGPR

   from staliro.staliro import staliro
   from staliro.options import Options

   # Define Signals and Specification
   benchmark = "AT1"
   results_folder = "Arch_Partx_Demo"

   AT1_phi = "G[0, 20] (speed <= 120)"
   specification = RTAMTDense(AT1_phi, {"speed": 0})
  
   signals = [
         SignalOptions(control_points = [(0, 100)]*7, signal_times=np.linspace(0.,50.,7)),
         SignalOptions(control_points = [(0, 325)]*3, signal_times=np.linspace(0.,50.,3)),
      ]

   MAX_BUDGET = 2000
   NUMBER_OF_MACRO_REPLICATIONS = 10
   
   model = AutotransModel()

   oracle_func = None
      
   optimizer = PartX(
            BENCHMARK_NAME=f"{benchmark}_budget_{MAX_BUDGET}_{NUMBER_OF_MACRO_REPLICATIONS}_reps",
            oracle_function = oracle_func,
            num_macro_reps = NUMBER_OF_MACRO_REPLICATIONS,
            init_budget = 20,
            bo_budget = 10,
            cs_budget = 20,
            n_tries_randomsampling = 1,
            n_tries_BO = 1
            alpha=0.05,
            R = 20,
            M = 500,
            delta=0.001,
            fv_quantiles_for_gp=[0.5,0.05,0.01],
            branching_factor = 2,
            uniform_partitioning = True,
            seed = 12345,
            gpr_model = InternalGPR(),
            bo_model = InternalBO(),
            init_sampling_type = "lhs_sampling",
            cs_sampling_type = "lhs_sampling",
            q_estim_sampling = "lhs_sampling",
            mc_integral_sampling_type = "uniform_sampling",
            results_sampling_type = "uniform_sampling",
            results_at_confidence = 0.95,
            results_folder_name = results_folder,
            num_cores = 1,
      )

   options = Options(runs=1, iterations=MAX_BUDGET, interval=(0, 50),  signals=signals)

   
   result = staliro(model, specification, optimizer, options)

