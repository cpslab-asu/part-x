import pickle
import numpy as np
import pickle
import unittest

from partx.sampling import lhs_sampling, uniform_sampling
from partx.utils import Fn, compute_robustness


class TestFunction(unittest.TestCase):
    def test1_uniform_sampling(self):
        with open("./tests/utils/goldResources/calRob_x_t1.pickle", "rb") as f:
            inputX = pickle.load(f)

        with open("./tests/utils/goldResources/calRob_y_t1.pickle", "rb") as f:
            inputY = pickle.load(f)

        def test_function(X):
            return X[0] ** 2 + X[1] ** 2

        decFunction = Fn(test_function)
        out = compute_robustness(inputX, decFunction)
        np.testing.assert_array_equal(inputY, out)

    def test2_compute_robustness(self):
        with open("./tests/utils/goldResources/calRob_x_t2.pickle", "rb") as f:
            inputX = pickle.load(f)

        with open("./tests/utils/goldResources/calRob_y_t2.pickle", "rb") as f:
            inputY = pickle.load(f)

        def test_function(X):
            return X[0] ** 2 + X[1] ** 2

        decFunction = Fn(test_function)
        out = compute_robustness(inputX, decFunction)
        np.testing.assert_array_equal(inputY, out)
