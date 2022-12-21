import numpy as np
import unittest
from partx.utils import branch_region

class TestBranchRegion(unittest.TestCase):
    def test1_branch_region(self):
        region_support = np.array([[-1., 1.], [-1., 2.]])
        direction_of_branching = 0
        uniform = True
        branching_factor = 2
        rng = np.random.default_rng(12345)
        new_reg = branch_region(region_support, direction_of_branching, uniform ,branching_factor, rng)

        np.testing.assert_array_equal(new_reg[0], np.array([[-1., 0.], [-1., 2.]]))
        np.testing.assert_array_equal(new_reg[1], np.array([[0., 1.], [-1., 2.]]))

    def test2_branch_region(self):
        region_support = np.array([[-1., 1.], [-1., 2.]])
        direction_of_branching = 0
        uniform = True
        branching_factor = 4
        rng = np.random.default_rng(12345)
        new_reg = branch_region(region_support, direction_of_branching, uniform ,branching_factor, rng)
        
        np.testing.assert_array_equal(new_reg[0], np.array([[-1., -0.5], [-1., 2.]]))
        np.testing.assert_array_equal(new_reg[1], np.array([[-0.5, 0], [-1., 2.]]))
        np.testing.assert_array_equal(new_reg[2], np.array([[0, 0.5], [-1., 2.]]))
        np.testing.assert_array_equal(new_reg[3], np.array([[0.5, 1], [-1., 2.]]))

    def test3_branch_region(self):
        region_support = np.array([[-1., 1.], [-1., 2.]])
        direction_of_branching = 1
        uniform = True
        branching_factor = 2
        rng = np.random.default_rng(12345)
        new_reg = branch_region(region_support, direction_of_branching, uniform ,branching_factor, rng)
        
        np.testing.assert_array_equal(new_reg[0], np.array([[-1., 1.], [-1., 0.5]]))
        np.testing.assert_array_equal(new_reg[1], np.array([[-1., 1.], [0.5, 2.]]))

    def test4_branch_region(self):
        region_support = np.array([[-1., 1.], [-1., 2.]])
        direction_of_branching = 1
        uniform = True
        branching_factor = 4
        rng = np.random.default_rng(12345)
        new_reg = branch_region(region_support, direction_of_branching, uniform ,branching_factor, rng)
        
        np.testing.assert_array_equal(new_reg[0], np.array([[-1., 1.], [-1., -0.25]]))
        np.testing.assert_array_equal(new_reg[1], np.array([[-1., 1.], [-0.25, 0.5]]))
        np.testing.assert_array_equal(new_reg[2], np.array([[-1., 1.], [0.5, 1.25]]))
        np.testing.assert_array_equal(new_reg[3], np.array([[-1., 1.], [1.25, 2]]))
        