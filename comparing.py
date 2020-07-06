"""
Module containing different functions to make results of similarity measures
comparable
"""

import pandas as pd
import numpy as np
from skimage import exposure

#Comparing functions
def binning_values_to_quantiles(map_array, num_bins=10):
    """
    Convert a map of values into n percentile bins.

    Each value on the map is replaced with the percentage bin it belongs to.
    0.3 means this value belongs to the 20%-30% bin which contains the 20%-30%
    smallest values of the map.

    All the bins have the same size

    Args:
        map_array (array): Map with values to scale
        num_bins (int): Number or bins

    Returns:
        Map with the bin numbers for each value
    """
    values = pd.DataFrame(np.array(map_array).flatten())
    bins = pd.qcut(values.iloc[:, 0], num_bins, labels=False)
    return np.array((bins + 1) / num_bins).reshape(map_array.shape)

def equalize_histogram(map_array, num_bins=10):
    """
    Scale a map of values using histogram equalization.

    It spreads out the most frequent intensity values, i.e. stretching out the
    value range of the map.

    Args:
        map_array (np.ndarray): Map with values to scale
        num_bins (int): Number or bins

    Returns:
        Map with the scaled values
    """
    return exposure.equalize_hist(map_array, nbins=num_bins)


def min_max_normalization(map_array, a=0, b=1):
    """
    Rescale a map of values ro range [a, b] using min-max normalization

    Args:
        map_array (np.ndarray): Map with values to scale
        a (int, optional): Lower bound
            Defaults to 0
        b (int, optional): Upper bound
            Defaults to 0

    Returns:
        Map with scaled values
    """
    min = map_array.min()
    max = map_array.max()

    return a + ((map_array - min) * (b - a) / (max - min))


#Preprocessing
def invert(measure):
    return (lambda x, y: - measure(x, y))
