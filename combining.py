"""
Module containing different functions to combine results of similarity measures.

All functions take in two numpy.ndarray and return one numpy.ndarray
"""
import numpy as np

#Combination functions
def sum(x, y):
    return x + y

def mult(x, y):
    return x * y

def max(x, y):
    return np.maximum(x, y)


#Preprocessing
def power_combination(combination_func):
    """
    Returns a function that combines the absolute values using combination_func

    Args:
        combination_func: function that combines two numpy.ndarray into one

    Returns:
        Function that combines absolute values using combination_func
    """
    return (lambda x, y: combination_func(abs(x), abs(y)))
