
from treelib import Tree


from partx.results import generate_statistics
import pathos, multiprocess
from pathos.multiprocessing import ProcessingPool as Pool
import dill
import pickle
import logging
import time
import datetime
import re, csv, itertools
import pathlib
import os
import numpy as np

folder_name = "Automatic_Transmission"
BENCHMARK_NAME = "AT6a_budget_5000"

quantiles_at = [0.01, 0.05, 0.5]
confidence_at = 0.95



result_dictionary = generate_statistics(BENCHMARK_NAME, 9, quantiles_at, confidence_at, folder_name)


print(result_dictionary)