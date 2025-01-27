import numpy as np
import unittest
from partx.utils import calculate_volume

class TestCalculateVolume(unittest.TestCase):
    def test1_calculate_volume(self):
        region_support = np.array([[-1., 1.], [-1., 2.]])
        assert calculate_volume(region_support) == 6
        

        