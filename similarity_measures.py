"""
Module containing different similarity measures for time series.
"""
import numpy as np
import scipy.spatial.distance as sc
from scipy.stats import spearmanr, kendalltau
import pyinform # pylint: disable=E0401
import minepy # pylint: disable=E0401
import similaritymeasures # pylint: disable=E0401
from sklearn.decomposition import PCA # pylint: disable=E0401
from rdc import rdc

def pearson_correlation(series1, series2):
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

def pearson_correlation_abs(series1, series2):
    """
    Compute the absolute Pearson correlation coefficient between two series

    Quantifies the degree of linear relationship between time series.

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Pearson correlation coefficient between the two series
    """
    return abs(pearson_correlation(series1, series2))

def spearman_correlation(series1, series2):
    """
    Compute the Spearman correlation coefficient between two series

    Benchmarks monotonic relationships between time series.

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Spearman correlation coefficient between the two series
    """
    return spearmanr(series1, series2).correlation

def kendall_tau(series1, series2):
    """
    Compute the Kendall Tau coefficient between two series

    Non-parametric measure of relationship between time series.

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Kendall Tau correlation coefficient between the two series
    """
    return kendalltau(series1, series2).correlation

def manhattan_distance(series1, series2):
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

def euclidean_distance(series1, series2):
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
    return 1 - sc.cosine(series1, series2)

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
    series1_2d = np.zeros((len(series1), 2))
    series1_2d[:, 0] = range(len(series1))
    series1_2d[:, 1] = series1

    series2_2d = np.zeros((len(series2), 2))
    series2_2d[:, 0] = range(len(series2))
    series2_2d[:, 1] = series2

    return similaritymeasures.dtw(series1_2d, series2_2d)[0]

def principal_component_distance(series1, series2, k=2):
    """
    Compute the distance of the first k principal components between two series

    Computes the difference between time series mapped into the first k PCs that
    explain the majority of the variance.

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series
        k (int): Number of Principal Components
            Defaults to 2

    Returns:
        Distance of values mapped into the first k principal components
    """
    series1_2d = np.zeros((len(series1), 2))
    series1_2d[:, 0] = range(len(series1))
    series1_2d[:, 1] = series1

    series2_2d = np.zeros((len(series2), 2))
    series2_2d[:, 0] = range(len(series2))
    series2_2d[:, 1] = series2

    pca1 = PCA().fit_transform(series1_2d)
    pca2 = PCA().fit_transform(series2_2d)

    distance = np.sqrt(np.sum(np.square(pca1[:, :k] - pca2[:, :k])))
    return distance

def maximal_information_coefficient(series1, series2):
    """
    Compute the maximal information coefficient between two series.

    MIC captures a wide range of associations both functional and not,
    and for functional relationships provides a score that roughly equals
    the coefficient of determination (R^2) of the data relative to the
    regression function

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Maximal information coefficient between the two series
    """
    mine = minepy.MINE()
    mine.compute_score(series1, series2)
    return mine.mic()

def randomized_dependence_coefficient(series1, series2):
    """
    Compute the randomized dependence coefficient between two series

    Measure of nonlinear dependence between random variables based on the
    Hirschfeld-Gebelein-Renyi Maximum Correlation Coefficient

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Randomized dependence coefficient between the two series
    """
    return rdc(np.array(series1), np.array(series2))

def distance_correlation(series1, series2):
    """
    Calculate the distance correlation introduced by Gábor J. Székely between two
    time series

    Args:
        series1 (numpy.ndarray): First series
        series2 (numpy.ndarray): Second series

    Returns:
        Distance Correlation between the two series

    Source:
        https://gist.github.com/satra/aa3d19a12b74e9ab7941

    """
    series1 = np.atleast_1d(series1)
    series2 = np.atleast_1d(series2)
    if np.prod(series1.shape) == len(series1):
        series1 = series1[:, None]
    if np.prod(series2.shape) == len(series2):
        series2 = series2[:, None]
    series1 = np.atleast_2d(series1)
    series2 = np.atleast_2d(series2)
    n = series1.shape[0]
    if series2.shape[0] != series1.shape[0]:
        raise ValueError('Number of samples must match')
    a = sc.squareform(sc.pdist(series1))
    b = sc.squareform(sc.pdist(series2))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def shift_to_positive(series):
    """
    Shift a series by adding the biggest negative value so all values are greater 0

    Args:
        series (numpy.ndarray): Series to shift

    Returns:
        Shifted series with all values greater 0
    """
    if min(series) >= 0:
        return series #No need to shift
    return series - min(series)

def normalize(series):
    """
    Normalize time series

    Args:
        series (np.ndarray): Time series to normalize

    Returns:
        Normalized time series
    """
    norm = np.linalg.norm(series)
    if norm == 0:
        return series
    return series / norm
