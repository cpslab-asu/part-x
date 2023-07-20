
Outputs
========

.. _reference_results_strcutrue:

Result Structure
-----------------

The results structure contains the following items:

- **fv_stats_wo_gp** 
   Pandas Dataframe consisting of the Falsification Volume calculated using just samples from classified, unclasssified and remaining regions.
   Typically, this dataframe will a 3*4 matrix and have the following structure.

   .. list-table:: 
      :widths: 25 25 25 25 25
      :header-rows: 1

      *  - 
         - Mean
         - Std_Error
         - LCB
         - UCB

      *  - Classified Region only 
         - xxx
         - xxx
         - xxx
         - xxx

      *  - Unclassified Regions only   
         - xxx
         - xxx
         - xxx
         - xxx
      
      *  - Classified + Unclassified Regions
         - xxx
         - xxx
         - xxx
         - xxx

..

- **fv_stats_with_gp**
   Pandas Dataframe consisting of the Falsification Volume calculated using generating gp from the samples and estimating these falsification volumes at certain confidence
   Typically, this dataframe will a N * 4 matrix and have the following structure.

   .. list-table:: 
      :widths: 25 25 25 25 25
      :header-rows: 1

      *  - Quantile
         - Mean
         - Std_Error
         - LCB
         - UCB

      *  - Quantile_1 
         - xxx
         - xxx
         - xxx
         - xxx

      *  - Quantile_2
         - xxx
         - xxx
         - xxx
         - xxx
      
      *  - ...
         - ...
         - ...
         - ...
         - ...
      
      *  - Quantile_N
         - xxx
         - xxx
         - xxx
         - xxx

..

- **falsified_true**
   Boolean values indicating if a replication had a falsifying output or not.

..

- **first_falsification_mean**
   Mean of the budget exhausted when the first falsification occured in every replication.

..

- **first_falsification_median**
   Median of the budget exhausted when the first falsification occured in every replication.

..

- **first_falsification_min**
   Minimum of budget exhausted when the first falsification occured in every replication.

..

- **first_falsification_max**
   Maximum of budget exhausted when the first falsification occured in every replication.

..

- **falsification_rate**
   The number of falsification occured over all macro-replications. If a falsification occurs in a single replication, that replication is said to have created a falsifying output. 

..

- **best_robustness**
   Best Robustness value obtained over all replications

..

- **first_falsification_points**
   The first falsification point in every replication

..   

- **best_falsification_points**
   Point corresponding to the best robustness value

..

- **non_falsification_points**
   The minimum robustenss points when no falsification has occured

..

Intermediate Files from Algorithm
----------------------------------

The code generates three folders, where every folder has different kind of files genenrated for every replication:

1) **BENCHMARK_NAME_log_files**

   This folder contains the information log of budget available, budget exhausted and the classified and unclassified regions.
   This information can often be used to understand the behaviour of how the algorithm behaved.

2) **BENCHMARK_NAME_result_generating_files**

   Here, the code generates the following files:
   
   - `BENCHMARK_NAME_options.pkl` : This file contains all the options that were defined for the experiments. 
   
   .. 

   - `BENCHMARK_NAME_all_result.pkl` : These files contains the raw values which are processed for geenrating the results.
   
   ..

   - `BENCHMARK_NAME_for_verif_result.pkl`: While generating the results for various quantiles, we often use the same values. These are basically arrays of falsified volumes of classified and unclssified regions and also contains the falsification volumes from GP for all the sub-regions.
   
   Then, the following files generated for every replication (note that XXXX here refers to the macro-replication number): 

   - `BENCHMARK_NAME_XXXX.pkl`: This file contains the tree structure and is the root of all information for a certain replications. Almost every statstic cna be obtained from the tree structure.
   
   .. 

   - `BENCHMARK_NAME_XXXX_point_history.pkl`: This file contains the history of points which were sampled and evaluated in the order by the algorithm.
   
   ..

   - `BENCHMARK_NAME_XXXX_fal_val_gp.pkl`: These are intermediate files for storing the values from gpr for every replication.
   ..

   - `BENCHMARK_NAME_XXXX_time.pkl`: These are intermediate files for storing the simulation, non-simulation, and total time for every replication.

3) **BENCHMARK_NAME_results_csv**
   Here, the code generates a csv file of the results for future use. These contains the same values as metnioned in :ref:`reference_results_strcutrue`
