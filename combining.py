"""
Module containing different functions to combine results of similarity measures.

All functions take in two numpy.ndarray and return one numpy.ndarray
"""
import numpy as np
from scipy.stats import entropy

#Combination functions
def sum(list_of_maps):
    return np.sum(list_of_maps, axis=0)

def mult(list_of_maps):
    result = np.ones_like(list_of_maps[0])
    for i in range(len(list_of_maps)):
        result = np.multiply(result, list_of_maps[i])
    return result

def max(list_of_maps):
    return np.max(list_of_maps, axis=0)

def mean(list_of_maps):
    return np.mean(list_of_maps, axis=0)

def median(list_of_maps):
    return np.median(list_of_maps, axis=0)

def min(list_of_maps):
    return np.min(list_of_maps, axis=0)

def std(list_of_maps):
    return np.std(list_of_maps, axis=0)

def entropy(list_of_maps):
    return entropy(list_of_maps)

def combine_power_with_sign(combination_func, list_of_maps, sign_map):
    """
    Returns a function that combines a list of similarity maps by taking the sign of sign_map
    and then combines all values in the list_of_maps by combining their absolute values with the combination_func

    Args:
        combination_func (function): function that combines a list of numpy.ndarray into one
        list_of_maps (list): List np.ndarray
        sign_map (np.ndarray): Array containing the sign values for every point

    Returns:
        Combination of a list of similarity maps by taking the sign of sign_map
        and then combines all values in the list_of_maps by combining their absolute values with the combination_func
    """
    strength = combination_func(list_of_maps)
    strength_with_sign = sign(sign_map) * strength
    return strength_with_sign


#Help funtions
def sign(x):
    """
    Variation of the np.sign function which returns 1 for x >= 0 and -1 for x < 0

    Args:
        x (numpy.ndarray): Array with values

    Returns:
        Array with signs of values
    """
    result = np.sign(x)
    np.where(result == 0, 1, result)
    return result
