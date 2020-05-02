"""
Module containing different similarity measures for time series
"""
import numpy as np
import scipy.spatial.distance as sc
from sklearn.feature_selection import mutual_info_classif

def correlation_similarity(series1, series2):
    """
    Compute the Pearson correlation coefficient between two series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Pearson correlation coefficient between the two series
    """
    return np.corrcoef([series1, series2])[0, 1]

def manhattan_similarity(series1, series2):
    """
    Compute the City Block (Manhattan) distance between two series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        City Block (Manhattan) distance coefficient between the two series
    """
    return sc.cityblock(series1, series2)

def mahalanobis_similarity(series1, series2):
    """
    Compute the Mahanalobis distance coefficient between two series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Mahalanobis distance between the two series
    """
    #TODO Implement Mahalanobis Distance
    return 0


def euclidean_similarity(series1, series2):
    """
    Compute the Euclidean distance between two series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Euclidean distance between the two series
    """
    return sc.euclidean(series1, series2)

def cosine_similarity(series1, series2):
    """
    Compute the Cosine distance between two series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Cosine distance between the two series
    """
    return sc.cosine(series1, series2)

def mutual_information(series1, series2):
    """
    Compute the Mutual Information between two series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Mutual Information between the two series
    """
    # TODO: Implement Mutual Information
    return 0

SIMILAIRITY_FUNCTIONS = {
    "correlation": correlation_similarity,
    "manhattan": manhattan_similarity,
    "mahalanobis": mahalanobis_similarity,
    "euclidean": euclidean_similarity,
    "cosine": cosine_similarity,
    "mutual_information": mutual_information
    }
