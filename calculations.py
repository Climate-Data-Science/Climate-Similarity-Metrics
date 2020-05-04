"""
    TODO: Module Docstring
"""

import numpy as np
from statsmodels.tsa.seasonal import STL # pylint: disable=E0401
import similarity_measures

def calculate_pointwise_similarity(map_array, lon, lat, level=0, sim_func="correlation"):
    """
    Calculate point-wise similarity of all points on a map to a reference point over time

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, latitude
        lon (int): Longitude of reference point
        lat (int): Latitude of reference point
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
        sim_func (str, optional): The similarity function that should be used.
            Defaults to Correlation Coefficient.
            Options: "correlation": Pearson's Correlation,
                     "manhattan": Mahattan Distance,
                     "mahalanobis": Mahalanobis Distance,
                     "euclidean": Euclidean Distance,
                     "cosine": Cosine Distance,
                     "mutual_information": Mutual Information,
                     "transfer_entropy": Transfer Entropy,
                     "relative_entropy": Relative Entropy

    Returns:
        2 dimensional numpy.ndarray with similarity values to reference point
    """
    len_time = map_array.shape[0]
    reference_series = np.array([map_array[time, level, lon, lat] for time in range(len_time)])
    return calculate_series_similarity(map_array, reference_series, level, sim_func)

def calculate_series_similarity(map_array, reference_series, level=0, sim_func="correlation"):
    """
    Calculate similarity of all points on a map to a reference series

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, latitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
        sim_func (str, optional): The similarity function that should be used.
            Defaults to Correlation Coefficient.
            Options: "correlation": Pearson's Correlation,
                     "manhattan": Mahattan Distance,
                     "mahalanobis": Mahalanobis Distance,
                     "euclidean": Euclidean Distance,
                     "cosine": Cosine Distance,
                     "mutual_information": Mutual Information
                     "transfer_entropy": Transfer Entropy,
                     "relative_entropy": Relative Entropy

    Returns:
        2 dimensional numpy.ndarray with similarity values to reference point
    """
    similarity = similarity_measures.SIMILARITY_FUNCTIONS[sim_func]
    map_array = map_array[:, level, :, :] #Eliminate level dimension
    (len_time, len_longitude, len_latitude) = map_array.shape
    sim = np.zeros((len_longitude, len_latitude))

    for lon_i in range(len_longitude):
        for lat_i in range(len_latitude):
            point_series = np.array([map_array[time, lon_i, lat_i] for time in range(len_time)])
            sim[lon_i, lat_i] = similarity(reference_series, point_series)

    return sim

def calculate_series_similarity_per_period(map_array, reference_series, level=0,
                                           period_length=12, sim_func="correlation"):
    """
    Calculate similarity of all points on a map to a reference series per period

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, latitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
        period_length(int, optional): Length of one period
            Defaults to 12
        sim_func (str, optional): The similarity function that should be used.
            Defaults to Correlation Coefficient.
            Options: "correlation": Pearson's Correlation,
                     "manhattan": Mahattan Distance,
                     "mahalanobis": Mahalanobis Distance,
                     "euclidean": Euclidean Distance,
                     "cosine": Cosine Distance,
                     "mutual_information": Mutual Information
                     "transfer_entropy": Transfer Entropy,
                     "relative_entropy": Relative Entropy

    Returns:
        List of similarity maps to reference series
    """
    len_time = map_array.shape[0]
    num_periods = int(round(len_time / period_length))
    sim = []
    for i in range(num_periods):
        period_similarity = calculate_series_similarity(map_array[i:i+period_length, :, :, :],
                                                        reference_series[i:i+period_length],
                                                        level,
                                                        sim_func)
        sim.append(period_similarity)
    return sim


def calculate_surrounding_mean(map_array, lon, lat, lon_step=0, lat_step=0):
    """
    Calculate Mean of the value at a point and of it's surrounding values

    Args:
        map_array (numpy.ndarray): Map with 2dimensions - longitude, latitude
        lon (int): Longitude of starting point
        lat (int): Latitude of starting point
        lon_step (int, optional): Stepsize in Longitude-dimension
            Defaults to 0
        lat_step (int, optional): Stepsize in Latitude-dimension
            Defaults to 0

    Returns:
        Mean of value at starting point with surrounding points
    """
    values = np.array(map_array[lon - lon_step : lon + lon_step + 1,
                                lat - lat_step: lat + lat_step + 1])
    return np.mean(values)



def deseasonalize_monthly_time_series(series):
    """
    Deseasonalize a monthly time series

    Args:
        series (numpy.ndarray): time series to deseasonalize
    Returns:
        numpy.ndarray containing the deseasonalized data
    """
    stl = STL(series, period=12)
    res = stl.fit()
    return res.observed - res.seasonal


def derive(map_array, lon, lat, level=0, lon_step=0, lat_step=0): # pylint: disable=R0913
    """
    Derive time series for a given index from a map.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, latitude
        lon (int): Longitude of starting point
        lat (int): Latitude of starting point
        level (int, optional): Level from which the index should be derived
            Defaults to 0
        lon_step (int, optional): Stepsize in Longitude-dimension:
            How many points in the horizontal direction should be taken into account.
            Defaults to 0
        lat_step (int, optional): Stepsize in Latitude-dimension:
            How many points in the vertical direction should be taken into account.
            Defaults to 0

    Returns:
        List containing the mean values of all values in the respective index per time
    """
    len_time = map_array.shape[0]

    time_series = []
    for time in range(len_time):
        value = calculate_surrounding_mean(map_array[time, level, :, :],
                                           lon, lat, lon_step, lat_step)
        time_series.append(value)

    return time_series
