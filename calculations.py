import numpy as np

def corr_similairity(series1, series2):
    return np.corrcoef([series1, series2])[0,1]

similairity_functions = {"corr": corr_similairity}

def pointwise_similairity(map_array, x0, y0, level, simFunct="corr"):
    """
    Caluclates point-wise similairity of all points on a map to a reference point over time

    Parameters:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, longitude, lattitude
        x0 (int): X-Component of reference point
        y0 (int): Y-Component of reference point
        level (int): Level on which the similairity should be caluculated
        simFunct (str): The similairity function that should be used. Default: Correlation Coefficient.
                            Options: "corr": Correlation Coefficient, more will follow

    Returns:
        sim (numpy.ndarray): 2 dimensional array with similairity values to reference point
    """

    similairity = similairity_functions[simFunct]
    map0 = map_array[:, level, :, :] #Eliminate level-dimension
    (lenT, lenX, lenY) = map0.shape
    sim = np.zeros((lenX, lenY))
    referenceSeries = np.array([map0[time, x0, y0] for time in range(lenT)])

    #Calculate similairity
    for x in range(lenX):
        for y in range(lenY):
            pointSeries = np.array([map0[time, x, y] for time in range(lenT)])
            sim[x,y] = similairity(referenceSeries, pointSeries)

    return sim
