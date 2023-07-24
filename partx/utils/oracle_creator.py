
from ..utils import calculate_volume
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from dataclasses import dataclass

@dataclass
class OracleResult:
    val: float
    sat: bool

class OracleCreator:
    def __init__(self, oracle_function, n_tries_randomsampling, n_tries_BO):

        """Helps to set up options for Part-X

        Args:
           
        """
        self.oracle_function = oracle_function
        self.n_tries_randomsampling = n_tries_randomsampling
        self.n_tries_BO = n_tries_BO

        
    def __call__(self, X):
        if self.oracle_function is not None:
            val = self.oracle_function(X)
            sat = val <= 0.0
        else:
            val = -np.inf
            sat = True
        return OracleResult(val, sat)
        