from ..utils import calculate_volume
import numpy as np

def classification(region_support: np.array, region_class:chr, min_volume:float, min_delta_q:list, max_delta_q:list)->chr:
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
    volume = calculate_volume(region_support)
    

    if volume <= min_volume:
        region_class = 'u'
    elif max_delta_q is None and min_delta_q is None:
        region_class = "i"
    elif region_class == "+":
        if max_delta_q <= 0:
            region_class = 'r+'
        else:
            region_class = '+'
    elif region_class == "-":
        if min_delta_q >= 0:
            region_class = 'r-'
        else:
            region_class = "-"
    elif region_class == 'r':
        if min_delta_q < 0:
            region_class = "-"
        elif max_delta_q > 0:
            region_class = "+"
        else:
            region_class = 'r'

    return region_class
