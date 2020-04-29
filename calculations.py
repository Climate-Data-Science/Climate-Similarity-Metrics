"""
    TODO: Module Docstring
"""

import numpy as np

def correlation_similarity(series1, series2):
    """
    Calculate the Pearson correlation coefficient between 2 series
    """
    return np.corrcoef([series1, series2])[0, 1]

SIMILAIRITY_FUNCTIONS = {
    "correlation": correlation_similarity}

def calculate_pointwise_similarity(map_array, lon, lat, level, sim_func="correlation"):
    """
    Calculate point-wise similarity of all points on a map to a reference point over time

    Parameters:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, latitude
        lon (int): Longitude of reference point
        lat (int): Latitude of reference point
        level (int): Level on which the similarity should be calculated
        sim_func (str): The similarity function that should be used.
            Default: Correlation Coefficient.
            Options: "corr": Correlation Coefficient, more will follow

    Returns:
        2 dimensional array with similarity values to reference point
    """
    len_time = map_array.shape[0]
    reference_series = np.array([map_array[time, level, lon, lat] for time in range(len_time)])
    return calculate_series_similarity(map_array, reference_series, level, sim_func)

def calculate_series_similarity(map_array, reference_series, level, sim_func="correlation"):
    """
    Calculate similarity of all points on a map to a reference series

    Parameters:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, latitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        level (int): Level on which the similarity should be calculated
        sim_func (str): The similarity function that should be used.
            Default: Correlation Coefficient.
            Options: "corr": Correlation Coefficient, more will follow

    Returns:
        sim (numpy.ndarray): 2 dimensional array with similarity values to reference point
    """
    similarity = SIMILAIRITY_FUNCTIONS[sim_func]
    map_array = map_array[:, level, :, :] #Eliminate level dimension
    (len_time, len_longitude, len_latitude) = map_array.shape
    sim = np.zeros((len_longitude, len_latitude))

    for lon_i in range(len_longitude):
        for lat_i in range(len_latitude):
            point_series = np.array([map_array[time, lon_i, lat_i] for time in range(len_time)])
            sim[lon_i, lat_i] = similarity(reference_series, point_series)

    return sim



def calculate_surrounding_mean(map_array, lon, lat, step=2):
    """
    Calculate Mean of the value at a point and of it's surrounding values

    Parameters:
        map_array (numpy.ndarray): Map with 2dimensions - longitude, latitude
        x0 (int): X-Component of starting point
        y0 (int): Y-Component of starting point
        step (int): Radius of values that will be take into account

    Returns:
        Mean of value at starting point with surrounding points
    """
    return np.mean(np.array(map_array[lon - step : lon + step + 1, lat - step: lat + step + 1]))

def derive_qbo(map_array, level=0):
    """
    Derive the QBO Index from a map

    Parameters:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, latitude
        level (int): Level from which the index should be derived

    Returns:
        qbo (list): QBO Index
    """
    lon = int(np.round((180 - 1) * (256 / 360)))
    lat = int(np.round((180 + 104) * (512 / 360)))

    #qbo = map_array[:, 0, x0, y0]

    qbo = [calculate_surrounding_mean(map_array[time, level, :, :], lon, lat, step=1)
           for time in range(len(map_array[:, level, 0, 0]))]

    return qbo
