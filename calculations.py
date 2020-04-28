import numpy as np

def corr_similarity(series1, series2):
    return np.corrcoef([series1, series2])[0,1]

similarity_functions = {"corr": corr_similarity}

def pointwise_similarity(map_array, x0, y0, level, simFunct="corr"):
    """
    Caluclates point-wise similarity of all points on a map to a reference point over time

    Parameters:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, lattitude
        x0 (int): X-Component of reference point
        y0 (int): Y-Component of reference point
        level (int): Level on which the similarity should be calculated
        simFunct (str): The similarity function that should be used. Default: Correlation Coefficient.
                            Options: "corr": Correlation Coefficient, more will follow

    Returns:
        sim (numpy.ndarray): 2 dimensional array with similarity values to reference point
    """
    lenT = map_array.shape[0]
    referenceSeries = np.array([map_array[time, level, x0, y0] for time in range(lenT)])
    sim = series_similarity(map_array, referenceSeries, level, simFunct)
    return sim

def series_similarity(map_array, referenceSeries, level, simFunct="corr"):
    """
    Caluclates similarity of all points on a map to a reference series

    Parameters:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, lattitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        level (int): Level on which the similarity should be calculated
        simFunct (str): The similarity function that should be used. Default: Correlation Coefficient.
                            Options: "corr": Correlation Coefficient, more will follow

    Returns:
        sim (numpy.ndarray): 2 dimensional array with similarity values to reference point
    """
    similarity = similarity_functions[simFunct]
    map0 = map_array[:, level, :, :] #Eliminate level dimension
    (lenT, lenX, lenY) = map0.shape
    sim = np.zeros((lenX, lenY))
    #Calculate similarity
    for x in range(lenX):
        for y in range(lenY):
            pointSeries = np.array([map0[time, x, y] for time in range(lenT)])
            sim[x,y] = similarity(referenceSeries, pointSeries)

    return sim



def calcmean(map_array, x0, y0, step=2):
    """
    Calculate Mean of the value at a point and of it's surrounding values

    Parameters:
        map_array (numpy.ndarray): Map with 2dimensions - longitude, lattitude
        x0 (int): X-Component of starting point
        y0 (int): Y-Component of starting point
        step (int): Radius of values that will be take into account

    Returns:
        Mean of value at starting point with surrounding points
    """
    return np.mean(np.array(map_array[x0 - step : x0 + step + 1, y0 - step: y0 + step + 1]))

def derive_QBO(map_array, level=0):
    """
    Derive the QBO Index from a map

    Parameters:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, lattitude
        level (int): Level from which the index should be derived

    Returns:
        qbo (list): QBO Index
    """
    x0 = int(np.round((180 - 1) * (256 / 360)))
    y0 = int(np.round((180 + 104) * (512 / 360)))

    #qbo = map_array[:, 0, x0, y0]

    qbo = [calcmean(map_array[time, level, :, :], x0, y0, step=1) for time in range(len(map_array[:, level, 0, 0]))]

    return qbo
