"""
Module containing different similarity measures for time series
"""
import numpy as np
import scipy.spatial.distance as sc
import pyinform
import similaritymeasures

def correlation_similarity(series1, series2):
    """
    Compute the Pearson correlation coefficient between two series

    Quantifies the degree of linear relationship between time series.

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

    Quantifies the absolute magnitude of the difference between time series.

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

    Quantifies the difference between time series but accounts for
    non-stationarity of variance and temporalcross-correlation.

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

    Quantifies the Euclidean distance of the difference between time series.

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

    Measure of the amount of mutual dependence between two random variables.

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

    Quantify information transfer between an information
    source and destination, conditioning out shared history effects.

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Transfer Entropy between the two series
    """
    return pyinform.transferentropy.transfer_entropy(shift_to_positive(series1),
                                                     shift_to_positive(series2),
                                                     k=2)

def conditional_entropy(series1, series2):
    """
    Compute the Relative Entropy between two series

    Measure of the amount of information required to describe a
    random variable series1 given knowledge of another random variable series2

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Relative Entropy between the two series
    """
    return pyinform.conditionalentropy.conditional_entropy(shift_to_positive(series1),
                                                           shift_to_positive(series2))

def dynamic_time_warping_distance(series1, series2):
    """
    Compute the Dynamic Time Warping distance between two series

    Dynamic time warping is an algorithm used to measure similarity between
    two sequences which may vary in time or speed.
    It works as follows:
        1. Divide the two series into equal points.
        2. Calculate the euclidean distance between the first point in the
            first series and every point in the second series. Store the minimum
            distance calculated. (this is the ‘time warp’ stage)
        3. Move to the second point and repeat 2. Move step by step along points
            and repeat 2 till all points are exhausted.
        4. Repeat 2 and 3 but with the second series as a reference point.
        5. Add up all the minimum distances that were stored and this is a
            true measure of similarity between the two series.

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Dynamic time warping distance between the two series
    """
    series1_2d = np.array([series1, range(len(series1))])
    series2_2d = np.array([series2, range(len(series2))])
    return similaritymeasures.dtw(series1_2d, series2_2d)[0]

def  principal_component_distance(series1, series2):
    """
    Compute the Principal Component distance between two series

    Quantifies the difference between time series PCs that explain
    the majority of the variance.

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Principal Component distance between the two series
    """
    ## TODO: Implement PCA

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

SIMILARITY_FUNCTIONS = {
    "correlation": correlation_similarity,
    "manhattan": manhattan_similarity,
    "mahalanobis": mahalanobis_similarity,
    "euclidean": euclidean_similarity,
    "cosine": cosine_similarity,
    "mutual_information": mutual_information,
    "transfer_entropy": transfer_entropy,
    "conditional_entropy": conditional_entropy,
    "dtw": dynamic_time_warping_distance,
    "pca": principal_component_distance,
    }
