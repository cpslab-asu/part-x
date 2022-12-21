import numpy as np
import numpy.typing as npt
from typing import List, Type
from .function import Fn


def compute_robustness(samples_in: npt.NDArray, test_function: Type[Fn]) -> npt.NDArray:
    """Compute the fitness (robustness) of the given sample.

    Args:
        samples_in: Samples points for which the fitness is to be computed.
        test_function: Test Function insitialized with Fn
    Returns:
        Fitness (robustness) of the given sample(s)
    """

    if samples_in.shape[0] == 1:
        samples_out = np.array([test_function(samples_in[0])])
    else:
        samples_out = np.apply_along_axis(
            lambda sample: test_function(sample), 1, samples_in
        )
    return samples_out
