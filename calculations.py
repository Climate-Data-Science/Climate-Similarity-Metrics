"""
    TODO: Module Docstring
"""

import numpy as np
import pandas as pd # pylint: disable=E0401
from joblib import Parallel, delayed # pylint: disable=E0401
import comparing as comp
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


def calculate_filtered_agreement_areas_threshold_combinations(map_array, reference_series, measures, value_thresholds,
                                                              agreement_thresholds, combination_func=np.mean,
                                                              agreement_func=np.std, filter_values_high=True,
                                                              filter_agreement_high=False,
                                                              scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Calculate the areas where the similarity measures agree on the dependencies.
    Contains the following steps:
        1. Compute similarity between reference series and map with every similarity measure
        2. Combine the similarity maps into two summary maps:
            - Combine using np.mean to get a summary value for the similarity measures
            - Combine using agreement_func to get an agreement value for the similarity measures
        3. Filter the maps using their respective thresholds
        4. Return map containing ones(point has satisfied both conditions) and zeros(not satisfied at least one condition).
        5. Repeat 3-4 for every combination of value thresholds and agreement thresholds

    Before the values are combined (Step 2), they are scaled with the scaling_func to make value ranges combinable.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List of similarity measures to compute similarity between two time series
        value_thresholds (List): List of thresholds to filter the combined similarity values on
        agreement_thresholds (List): List of thresholds to filter the agreement on
        combination_func (function, optional): Function to compute the combined value between similarity measures
            Defaults to np.mean
        agreement_func (function, optional): Function to compute agreement between similarity values
            Defaults to np.std
        filter_values_high (Boolean, optional): Boolean indicating if combined similarity values should be
                                                filtered high (if set to True) or low (if set to False)
            Defaults to True
        filter_agreement_high (Boolean, optional): Boolean indicating if agreement values should be
                                                   filtered high (if set to True) or low (if set to False)
            Defaults to False
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0

    Returns:
        Array with the resulting agreement maps with the following dimensions (value_threshold, agreement_threshold, latitude, longitude)
    """
    maps = np.zeros((len(value_thresholds), len(agreement_thresholds), len(map_array[0,0,:,0]), len(map_array[0,0,0,:])))
    similarities = []
    mean_map = np.zeros(map_array[0, 0, :, :].shape)
    agreement = np.zeros_like(mean_map)

    for measure in measures:
        similarity = calculate_series_similarity(map_array, reference_series, level, measure)
        similarity = scaling_func(similarity)
        similarities.append(similarity)

    agreement = agreement_func(similarities, axis=0)
    mean_map = combination_func(similarities, axis=0)


    for i, value_threshold in enumerate(value_thresholds):
        for j, agreement_threshold in enumerate(agreement_thresholds):
            map = np.ones_like(mean_map)

            agreement_filtered = filter_map(agreement, agreement_threshold, high=filter_agreement_high)
            mean_map_filtered = filter_map(mean_map, value_threshold, high=filter_values_high)

            map = map * agreement_filtered * mean_map_filtered
            maps[i, j, :, :] = map

    return maps


def calculate_filtered_agreement_areas(map_array, reference_series, measures, value_threshold, agreement_threshold,
                                       combination_func=np.mean, agreement_func=np.std, filter_values_high=True,
                                       filter_agreement_high=False, scaling_func=comp.binning_values_to_quantiles,
                                       level=0):
    """
    Calculate the areas where the similarity measures agree on the dependencies.
    Contains the following steps:
        1. Compute similarity between reference series and map with every similarity measure
        2. Combine the similarity maps into two summary maps:
            - Combine using np.mean to get a summary value for the similarity measures
            - Combine using agreement_func to get an agreement value for the similarity measures
        3. Filter the maps using their respective thresholds
        4. Return map containing ones(point has satisfied both conditions) and zeros(not satisfied at least one condition).

    Before the values are combined (Step 2), they are scaled with the scaling_func to make value ranges combinable.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List of similarity measures to compute similarity between two time series
        value_threshold (Float): Threshold to filter the combined similarity values on
        agreement_threshold (Float): Threshold to filter the agreement on
        combination_func (function, optional): Function to compute the combined value between similarity measures
            Defaults to np.mean
        agreement_func (function, optional): Function to compute agreement between similarity values
            Defaults to np.std
        filter_values_high (Boolean, optional): Boolean indicating if combined similarity values should be
                                                filtered high (if set to True) or low (if set to False)
            Defaults to True
        filter_agreement_high (Boolean, optional): Boolean indicating if agreement values should be
                                                   filtered high (if set to True) or low (if set to False)
            Defaults to False
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0

    Returns:
        Agreement map containing ones (points that satisfy conditions) and
        zeros (points that don't satisfy at least one condition)
    """
    similarities = []
    mean_map = np.zeros(map_array[0, 0, :, :].shape)
    agreement = np.zeros_like(mean_map)

    for measure in measures:
        similarity = calculate_series_similarity(map_array, reference_series, level, measure)
        similarity = scaling_func(similarity)
        similarities.append(similarity)

    agreement = agreement_func(similarities, axis=0)
    mean_map = combination_func(similarities, axis=0)

    map = np.ones_like(mean_map)
    agreement_filtered = filter_map(agreement, agreement_threshold, high=filter_agreement_high)
    mean_map_filtered = filter_map(mean_map, value_threshold, high=filter_values_high)
    map = map * agreement_filtered * mean_map_filtered

    return map


def filter_map(map, threshold, high=True):
    """
    Filter a map of values based on a threshold.

    Args:
        map (array): Array of value to be filtered
        threshold (float): Threshold indicating which values to keep
        high (Boolean, optional): If set to True then values >= threshold are kept,
                                  else values < threshold are kept
            Defaults to True
    """
    filtered_map = None
    if high:
        filtered_map = (map >= threshold)
    else:
        filtered_map = (map < threshold)
    return filtered_map


def apply_mask_on_map(map, mask):
    """
    Apply a binary mask on a map of values.

    Args:
        Map (array): Map containing values
        Mask (array): Array containing ones for the points to keep and zeros for the points to drop

    Returns:
        Masked map
    """
    map_array = np.array(map)
    mask_array = np.array(mask)
    return np.ma.masked_array(map_array, np.logical_not(mask_array))
