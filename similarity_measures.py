"""
Module containing different similarity measures for time series
"""
import numpy as np
import scipy.spatial.distance as sc
import pyinform

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
    return pyinform.mutualinfo.mutual_info(shift_to_positive(series1),
                                           shift_to_positive(series2))

def transfer_entropy(series1, series2):
    """
    Compute the Transfer Entropy between two series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Transfer Entropy between the two series
    """
    return pyinform.transferentropy.transfer_entropy(shift_to_positive(series1),
                                                     shift_to_positive(series2),
                                                     k=2)

def relative_entropy(series1, series2):
    """
    Compute the Relative Entropy between two series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Relative Entropy between the two series
    """
    return pyinform.relativeentropy.relative_entropy(series1, series2)

def shift_to_positive(series):
    """
    Shifts a series by adding the biggest negative value so all values are greater 0

    Args:
        series (numpy.ndarray): Series to shift

    Returns:
        Shifted series with all values greater 0
    """
    if min(series) >= 0:
        return series #No need to shift
    else:
        return series - min(series)

SIMILAIRITY_FUNCTIONS = {
    "correlation": correlation_similarity,
    "manhattan": manhattan_similarity,
    "mahalanobis": mahalanobis_similarity,
    "euclidean": euclidean_similarity,
    "cosine": cosine_similarity,
    "mutual_information": mutual_information,
    "transfer_entropy": transfer_entropy,
    "relative_entropy": relative_entropy
    }
