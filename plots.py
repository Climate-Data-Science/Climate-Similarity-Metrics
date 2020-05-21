"""
    TODO: Module Docstring
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import calculations as calc

months = ["January", "February", "March", "April", "May",
          "June", "July", "August", "September", "October",
          "November", "December"]

def plot_similarities(map_array, reference_series, metrics, level=0, mode="whole_period"):
    """
    Plot the similarity of a reference data series and all points on the map regarding different
    similarity measures

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        metrics (list): List with similarity metrics to compute similarity between two time series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
        mode (str, optional): Mode of visualization
            Options: "whole_period": Similarity over whole period
                     "whole_period_per_month": Similarity over whole period, every month seperately
                     "whole_period_winter_only": Similarity over whole period, only winter months
            Defaults to "whole_period"
    """
    if mode == "whole_period":
        plot_similarities_whole_period(map_array, reference_series, metrics, level)
    elif mode == "whole_period_per_month":
        plot_similarities_whole_period_per_month(map_array, reference_series, metrics, level)
    elif mode == "whole_period_winter_only":
        plot_similarities_winter_only(map_array, reference_series, metrics, level)
    else:
        print("Mode not available")

def plot_similarities_whole_period(map_array, reference_series, metrics, level=0):
    """
    Plot the similarity of a reference data series and all points on the map for the whole period
    regarding different similarity measures

    Each column contains a different similarity metric.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        metrics (list): List with similarity metrics to compute similarity between two time series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    fig, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=(8*len(metrics), 10))

    for i, metric in enumerate(metrics):
        #Compute similarity
        sim_whole_period = calc.calculate_series_similarity(map_array,
                                                            reference_series,
                                                            level,
                                                            metric)

        #Draw map
        m = Basemap(projection='mill', lon_0=30, resolution='l', ax=ax[i])
        m.drawcoastlines()
        lons, lats = m.makegrid(512, 256)
        x, y = m(lons, lats)

        #Draw similarity
        cs = m.contourf(x, y, sim_whole_period[:, :])
        cbar = m.colorbar(cs, location='bottom', pad="5%")

        ax[i].set_title(metric.__name__)

    fig.suptitle("Similarity between QBO and all other points 1979 - 2019")
    plt.show()


def plot_similarities_whole_period_per_month(map_array, reference_series, metrics, level=0):
    """
    Plot the similarity of a reference data series and all points on the map for the whole period,
    but every month seperately, regarding different similarity measures

    Each column contains a different similarity metric and each row contains a different month.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        metrics (list): List with similarity metrics to compute similarity between two time series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    fig, ax = plt.subplots(figsize=(8*len(metrics), 14*len(metrics)), nrows=12, ncols=len(metrics))
    fig.subplots_adjust(hspace=0, wspace=0)

    for month in range(len(months)):
        ax[month][0].set_ylabel(months[month])

        #Extract monthly values
        map_array_month = np.array([map_array[12 * i + month, :, :, :] for i in range(40)])
        reference_series_month = [reference_series[12 * i + month] for i in range(40)]

        for i, metric in enumerate(metrics):
            ax[0][i].set_title(metric.__name__)

            #Calculate similarities
            similarity_month = calc.calculate_series_similarity(map_array_month,
                                                                reference_series_month,
                                                                level,
                                                                metric)

            #Plot Map
            m = Basemap(projection='mill', lon_0=30, resolution='l', ax=ax[month][i])
            m.drawcoastlines()
            lons, lats = m.makegrid(512, 256)
            x, y = m(lons, lats)
            cs = m.contourf(x, y, similarity_month)

    fig.suptitle("Similarity between QBO and all other points 1979 - 2019 per month")
    plt.show()

def plot_similarities_winter_only(map_array, reference_series, metrics, level=0):
    """
    Plot the similarity of a reference data series and all points on the map for the whole
    period, but only winter months are taken into account, regarding different similarity
    measures

    Each column contains a different similarity metric.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        metrics (list): List with similarity metrics to compute similarity between two time series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    fig, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=(8*len(metrics), 10))

    winter_indices = []
    for i in range(40):
        year = 12 * i
        winter_indices.append(year) #January
        winter_indices.append(year + 1) #February
        winter_indices.append(year + 11) #December

    #Extract winter values
    reference_series_winter = reference_series[winter_indices]
    map_array_winter = map_array[winter_indices, :, :, :]

    for i, metric in enumerate(metrics):
        #Compute similarity
        sim_whole_period_winter = calc.calculate_series_similarity(map_array_winter,
                                                                   reference_series_winter,
                                                                   level,
                                                                   metric)

        #Draw map
        m = Basemap(projection='mill', lon_0=30, resolution='l', ax=ax[i])
        m.drawcoastlines()
        lons, lats = m.makegrid(512, 256)
        x, y = m(lons, lats)

        #Draw similarity
        cs = m.contourf(x, y, sim_whole_period_winter[:, :])
        cbar = m.colorbar(cs, location='bottom', pad="5%")

        ax[i].set_title(metric.__name__)

    fig.suptitle("Similarity between QBO and all other points 1979 - 2019 for Winter months")
    plt.show()
