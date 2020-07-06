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

def mean(x, y):
    return (x + y) / 2

def take_sign_first_value_second(x, y):
    return sign(x) * abs(y)

def take_sign_second_value_first(x, y):
    return take_sign_first_value_second(y, x)


#Preprocessing
def power_combination(combination_func):
    """
    Returns a function that combines the absolute values using combination_func

    Args:
        combination_func (function): function that combines two numpy.ndarray into one

    Returns:
        Function that combines absolute values using combination_func
    """
    return (lambda x, y: combination_func(abs(x), abs(y)))


def take_sign_first_strength_both(combination_func):
    """
    Returns a function that combines two values by taking the sign of the first value
    and then combines the two values by combining their absolute values with the combination_func

    Args:
        combination_func (function): function that combines two numpy.ndarray into one

    Returns:
        Function that combines two values by taking the sign of the first value
        and combining the two absolute values with the combination_func
    """
    return (lambda x, y: sign(x) * combination_func(abs(x), abs(y)))


def take_sign_second_strength_both(combination_func):
    """
    Returns a function that combines two values by taking the sign of the second value
    and then combines the two values by combining their absolute values with the combination_func

    Args:
        combination_func (function): function that combines two numpy.ndarray into one

    Returns:
        Function that combines two values by taking the sign of the first value
        and combining the two absolute values with the combination_func
    """
    return (lambda x, y: sign(y) * combination_func(abs(x), abs(y)))


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
    np.where(result==0, 1, result)
    return result
