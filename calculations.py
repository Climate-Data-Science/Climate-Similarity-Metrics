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
