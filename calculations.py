"""
    TODO: Module Docstring
"""

import numpy as np
import pandas as pd # pylint: disable=E0401
from joblib import Parallel, delayed # pylint: disable=E0401

import similarity_measures

def calculate_pointwise_similarity(map_array, lat, lon, level=0,
                                   sim_func=similarity_measures.pearson_correlation):
    """
    Calculate point-wise similarity of all points on a map to a reference point over time

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        lat (int): Latitude of reference point
        lon (int): Longitude of reference point
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
        sim_func (str, optional): The similarity function that should be used.
            Defaults to Pearson's Correlation Coefficient.

    Returns:
        2 dimensional numpy.ndarray with similarity values to reference point
    """
    len_time = map_array.shape[0]
    reference_series = np.array([map_array[time, level, lat, lon] for time in range(len_time)])
    return calculate_series_similarity(map_array, reference_series, level, sim_func)


def calculate_series_similarity(map_array, reference_series, level=0,
                                sim_func=similarity_measures.pearson_correlation):
    """
    Calculate similarity of all points on a map to a reference series

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
        sim_func (str, optional): The similarity function that should be used.
            Defaults to Pearon's Correlation Coefficient.

    Returns:
        2 dimensional numpy.ndarray with similarity values to reference point
    """
    map_array = map_array[:, level, :, :] #Eliminate level dimension
    (len_latitude, len_longitude) = map_array.shape[1:]
    sim = np.zeros((len_latitude, len_longitude))

    sim[:, :] = Parallel(n_jobs=-1)(delayed(calculate_series_similarity_on_latitude)
                                    (map_array[:, lat, :], reference_series, sim_func)
                                    for lat in range(len_latitude))

    return np.array(sim).reshape(len_latitude, len_longitude)


def calculate_series_similarity_on_latitude(map_array, reference_series,
                                            sim_func=similarity_measures.pearson_correlation):
    """
    Calculate similarity of all points on a specific latitude to a reference series

    Args:
        map_array (numpy.ndarray): Map with 2 dimensions - time, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        sim_func (str, optional): The similarity function that should be used.
            Defaults to Pearon's Correlation Coefficient.

    Returns:
        1 dimensional numpy.ndarray with similarity values of points on this latitude
        to reference point
    """
    sim = np.zeros(map_array.shape[1])
    for i in range(map_array.shape[1]):
        sim[i] = sim_func(map_array[:, i], reference_series)
    return sim

def calculate_series_similarity_per_period(map_array, reference_series,
                                           level=0, period_length=12,
                                           sim_func=similarity_measures.pearson_correlation):
    """
    Calculate similarity of all points on a map to a reference series per period

    If the length of the series is no multiple of the period length, values from behind will be
    dropped until this condition is met.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
        period_length(int, optional): Length of one period
            Defaults to 12
        sim_func (str, optional): The similarity function that should be used.
            Defaults to Pearson's Correlation Coefficient.

    Returns:
        List of similarity maps to reference series
    """
    len_time = map_array.shape[0]
    num_periods = int(np.floor(len_time / period_length))
    sim = []
    for i in range(num_periods):
        start = i * period_length
        end = start + period_length
        period_similarity = calculate_series_similarity(map_array[start:end, :, :, :],
                                                        reference_series[start:end],
                                                        level,
                                                        sim_func)
        sim.append(period_similarity)
    return sim


def calculate_surrounding_mean(map_array, lat, lon, lat_step=0, lon_step=0):
    """
    Calculate Mean of the value at a point and of it's surrounding values

    Args:
        map_array (numpy.ndarray): Map with 2dimensions - latitude, longitude
        lat (int): Latitude of starting point
        lon (int): Longitude of starting point
        lat_step (int, optional): Stepsize in Latitude-dimension
            Defaults to 0
        lon_step (int, optional): Stepsize in Longitude-dimension
            Defaults to 0

    Returns:
        Mean of value at starting point with surrounding points
    """
    values = np.array(map_array[lat - lat_step: lat + lat_step + 1,
                                lon - lon_step: lon + lon_step + 1])
    return np.mean(values)


def deseasonalize_map(map_array, period_length=12):
    """
    Deseasonalize every data point of a map by subtracting the respective mean and dividing by
    the respective standard deviation.

    For example: Monthly (period_length = 12). From each value, subtract the alltime
    mean for this month and divide by the alltime standard deviation for this month.

    If the length of the time dimension is no multiple of the period length, values from behind
    will be dropped until this condition is met.

    Args:
        map_array (np.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        period_length (int): length of one period
            Defaults to 12

    Returns:
        Deseasonalized map
    """
    (len_time, len_level, len_latitude, len_longitude) = map_array.shape
    num_periods = int(np.floor(len_time / period_length))

    reshaped_map_array = map_array[:num_periods * period_length, :, :, :].reshape(num_periods,
                                                                                  period_length,
                                                                                  len_level,
                                                                                  len_latitude,
                                                                                  len_longitude)

    period_mean = np.mean(reshaped_map_array, axis=0)
    period_std = np.std(reshaped_map_array, axis=0)

    deseasonalized_map = (reshaped_map_array - period_mean) / period_std

    return deseasonalized_map.reshape(num_periods * period_length,
                                      len_level,
                                      len_latitude,
                                      len_longitude)


def deseasonalize_time_series(series, period_length=12):
    """
    Deseasonalize a time series by subtracting the respective mean and dividing by the respective
    standard deviation.

    For example: Monthly time series. From each value, subtract the alltime mean for this month and
    divide by the alltime standard deviation for this month.

    If the length of the series is no multiple of the period length, values from behind will be
    dropped until this condition is met.

    Args:
        series (np.ndarray): time series to deseasonalize
        period_length (int): length of one period
            Defaults to 12

    Returns:
        Deseasonalized time series
    """
    period_mean = np.zeros(period_length)
    period_std = np.zeros(period_length)

    len_time = len(series)
    num_periods = int(np.floor(len_time / period_length))

    for i in range(period_length):
        period_mean[i] = np.mean(np.array([series[j * period_length + i]
                                           for j in range(num_periods)]))
        period_std[i] = np.std(np.array([series[j * period_length + i]
                                         for j in range(num_periods)]))

    #Cut off last values that prevent series from having length that is a multiple of period_length
    series_short = series[:num_periods*period_length]

    normalized_series = [(series_short[k] - period_mean[k % period_length])
                         / period_std[k % period_length] for k in range(len(series_short))]
    return normalized_series


def derive(map_array, lat, lon, level=0, lat_step=0, lon_step=0): # pylint: disable=R0913
    """
    Derive time series for a given index from a map.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        lat (int): Latitude of starting point
        lon (int): Longitude of starting point
        level (int, optional): Level from which the index should be derived
            Defaults to 0
        lat_step (int, optional): Stepsize in Latitude-dimension:
            How many points in the vertical direction should be taken into account.
            Defaults to 0
        lon_step (int, optional): Stepsize in Longitude-dimension:
            How many points in the horizontal direction should be taken into account.
            Defaults to 0

    Returns:
        List containing the mean values of all values in the respective index per time
    """
    len_time = map_array.shape[0]

    time_series = []
    for time in range(len_time):
        value = calculate_surrounding_mean(map_array[time, level, :, :],
                                           lat, lon, lat_step, lon_step)
        time_series.append(value)

    return time_series


def convert_coordinates_to_grid(geo_coordinates, value):
    """
    Converts geographical coordinates into indices for values stored in an N128 Gaussian Grid system

    Args:
        geo_coordinates (List): List containing the meaning of the indices expressed
                                in geographical coordinates
        value (int): Coordinate to convert

    Returns:
        Indice for the respective geographical coordinate
    """
    tolerance = 0.1
    while len(np.where((geo_coordinates < value + tolerance) &
                       (geo_coordinates > value - tolerance))[0]) == 0:
        tolerance += 0.1

    gridpoint = np.where((geo_coordinates < value + tolerance) &
                         (geo_coordinates > value - tolerance))[0][0]

    return gridpoint


def binning_values_to_quantiles(map_array, num_bins=10):
    """
    Convert a map of values into n percentile bins.

    Each value on the map is replaced with the percentage bin it belongs to.
    0.3 means this value belongs to the 20%-30% bin which contains the 20%-30%
    smallest values of the map.

    All the bins have the same size

    Args:
        map_array (array): Map with values to convert
        num_bins (int): Number or bins

    Returns:
        Map with the bin numbers for each value
    """
    values = pd.DataFrame(np.array(map_array).flatten())
    bins = pd.qcut(values.iloc[:, 0], num_bins, labels=False)
    return np.array((bins + 1) / num_bins).reshape(map_array.shape)
