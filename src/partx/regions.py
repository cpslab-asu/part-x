from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import attrs
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

RegionClass: TypeAlias = Literal["u", "i", "+", "-", "r", "r+", "r-"]


def _region_volume(support: NDArray) -> float:
    return np.prod(support[:, 1] - support[:, 0], axis = 0)


def _branch_region(
    support: NDArray,
    direction: int,
    factor: int,
    rng: Generator,
    *,
    uniform: bool,
) -> list[NDArray]:
    dim_length = support[direction][1] - support[direction][0]

    if uniform:
        split_array = support[direction][0] + (np.arange(factor+1) / factor) * dim_length
    else: 
        split_array = support[direction][0] + np.sort(np.insert(rng.uniform(0, 1, factor - 1), 0, [0,1])) * dim_length
        # print(split_array)
    
    new_bounds = []

    for i in range(factor):
        temp = support.copy()
        temp[direction, 0] = split_array[i]
        temp[direction, 1] = split_array[i+1]
        
        new_bounds.append(temp)

    return new_bounds


def _classify_region(
    support: NDArray,
    prior_class: RegionClass,
    min_volume: float,
    min_delta_q: float,
    max_delta_q:float,
) -> RegionClass:
    """Function classifies the region based on the its Quantile estimates

    Args:
        region_support: The bounds of a region.
        region_class: The class of each region in previous iteration
        min_volume: Minimum Volume threshold for Classification
        lower_bound: lower bound of quantile estimates
        upper_bound: upper bound of quantile estimates


    Returns:
        chr: list of regions with corresponding class (Unidentified,Plus,Minus,RPlus,RMinus,Rem)
    """
    volume = _region_volume(support)
    

    if volume <= min_volume:
        return 'u'
    elif max_delta_q is None and min_delta_q is None:
        return "i"
    elif prior_class == "+":
        if max_delta_q <= 0:
            return 'r+'
        else:
            return '+'
    elif prior_class == "-":
        if min_delta_q >= 0:
            return 'r-'
        else:
            return "-"
    elif prior_class == 'r':
        if min_delta_q < 0:
            return "-"
        elif max_delta_q > 0:
            return "+"
        else:
            return 'r'

    raise RuntimeError()


@attrs.define()
class Region:
    """Area of the sampling space defined as a hypercube.

    Args:
        support: The bounds of the region within which the sampling is to be done. Region bounds
            must be given as an N x 2 matrix, were N is the dimensionality of the test function.
        rng: The random number generator to be used for non-uniform branching
    """

    support: NDArray
    rng: Generator

    @property
    def volume(self) -> float:
        """Volume of the region defined by a hypercube."""

        return _region_volume(self.support)

    def branch(self, direction: int, factor: int, *, uniform: bool) -> list[Region]:
        """Generate new region supports based on direction of branching and the branching factor.

        For now, the partitioning of space is uniformly done.

        Args:
            direction: Dimension or direction in which branching is to be done
            factor: How many new region supports to be made from the previous region support
            uniform: Do we partition the region uniformly or randomly. If True, uniform partition takes place

        Returns:
            regions: The new regions branched from the original
        """

        new_supports =  _branch_region(self.support, direction, factor, self.rng, uniform=uniform)
        return [attrs.evolve(self, support=support) for support in new_supports]

    def classify(
        self,
        prior_class: RegionClass,
        min_volume: float,
        delta_q_range: tuple[float, float]
    ) -> RegionClass:
        """Classify a region for sample allocation.

        Args:
            prior_class: The previous classification of the region, if any
            min_volume: The minimum volume of the region
            delta_q_range: The upper and lower bounds of the range of delta_q values

        Returns:
            class: The new classification of the region
        """

        return _classify_region(self.support, prior_class, min_volume, delta_q_range[0], delta_q_range[1])

