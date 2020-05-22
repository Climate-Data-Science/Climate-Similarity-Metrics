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


def plot_similarity_dependency(map_array, reference_series, metric1, metric2, level=0):
    """
    Calculate and plot dependency between two similarity metrics with one similarity
    metric on the x-axis and one on the y-axis

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        metric1 (function): First similarity metric to compute similarity between two time series
        metric2 (function): Second similarity metric to compute similarity between two time series
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    #Compute similarities
    sim_metric1 = calc.calculate_series_similarity(map_array, reference_series, level, metric1)
    sim_metric2 = calc.calculate_series_similarity(map_array, reference_series, level, metric2)

    #Plot dependency
    plt.scatter(sim_metric1, sim_metric2, )
    plt.xlabel(metric1.__name__)
    plt.ylabel(metric2.__name__)
    plt.title("Dependency between {} and {}".format(metric1.__name__, metric2.__name__))
    plt.show()


def plot_similarity_dependency_regions(map_array, reference_series, metric1, metric2,
                                       mode="high_high", level=0):
    """
    Plot regions where two similarity metrics have extreme values.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        metric1 (function): First similarity metric to compute similarity between two time series
        metric2 (function): Second similarity metric to compute similarity between two time series
        mode (str, optional): Mode defining which extremes to visualize
            Options: "high_high": High values in metric1 and high values in metric2
                     "high_low": High values in metric1 and low values in metric2
                     "low_high": Low values in metric1 and high values in metric2
                     "low_low": Low values in metric1 and low values in metric2
            Defaults to "high_high"
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    #Compute similarities
    sim_metric1 = calc.calculate_series_similarity(map_array, reference_series, level, metric1)
    sim_metric2 = calc.calculate_series_similarity(map_array, reference_series, level, metric2)

    if mode == "high_high":
        plot_high_high_similarity_dependency_regions(sim_metric1, sim_metric2)
        plt.title("High values in {} and high values in {}".format(metric1.__name__,
                                                                   metric2.__name__))
        plt.show()
    elif mode == "high_low":
        plot_high_low_similarity_dependency_regions(sim_metric1, sim_metric2)
        plt.title("High values in {} and low values in {}".format(metric1.__name__,
                                                                  metric2.__name__))
        plt.show()
    elif mode == "low_high":
        plot_high_low_similarity_dependency_regions(sim_metric2, sim_metric1)
        plt.title("Low values in {} and high values in {}".format(metric1.__name__,
                                                                  metric2.__name__))
        plt.show()
    elif mode == "low_low":
        plot_low_low_similarity_dependency_regions(sim_metric1, sim_metric2)
        plt.title("Low values in {} and low values in {}".format(metric1.__name__,
                                                                 metric2.__name__))
        plt.show()
    else:
        print("Mode unavailable")


def plot_high_high_similarity_dependency_regions(sim_metric1, sim_metric2):
    """
    Plot points where similarity values for both similarity metrics are high

    Args:
        sim_metric1 (np.ndarray): Map with shape (256, 512) containing similarity values
        sim_metric1 (np.ndarray): Map with shape (256, 512) containing similarity values
    """
    values1 = sim_metric1 > np.percentile(sim_metric1, 95)
    values2 = sim_metric2 > np.percentile(sim_metric2, 95)
    indexes_to_draw = values1 & values2

    draw_indexes_on_map(indexes_to_draw)


def plot_high_low_similarity_dependency_regions(sim_metric1, sim_metric2):
    """
    Plot points where similarity values are high for first similarity metric and low for
    the second similarity metric

    Args:
        sim_metric1 (np.ndarray): Map with shape (256, 512) containing similarity values
        sim_metric1 (np.ndarray): Map with shape (256, 512) containing similarity values
    """
    values1 = sim_metric1 > np.percentile(sim_metric1, 95)
    values2 = sim_metric2 < np.percentile(sim_metric2, 5)
    indexes_to_draw = values1 & values2

    draw_indexes_on_map(indexes_to_draw)


def plot_low_low_similarity_dependency_regions(sim_metric1, sim_metric2):
    """
    Plot points where similarity values for both similarity metrics are low

    Args:
        sim_metric1 (np.ndarray): Map with shape (256, 512) containing similarity values
        sim_metric1 (np.ndarray): Map with shape (256, 512) containing similarity values
    """
    values1 = sim_metric1 < np.percentile(sim_metric1, 5)
    values2 = sim_metric2 < np.percentile(sim_metric2, 5)
    indexes_to_draw = values1 & values2

    draw_indexes_on_map(indexes_to_draw)


def draw_indexes_on_map(indexes_to_draw):
    """
    Draw points on maps

    Args:
        indexes_to_draw (np.ndarray): Array of shape (256, 512) containing Boolean values
    """
    #Draw map
    m = Basemap(projection='mill', lon_0=30, resolution='l')
    m.drawcoastlines()
    lons, lats = m.makegrid(512, 256)
    x, y = m(lons, lats)

    #Mark points on map
    cs = m.contourf(x, y, indexes_to_draw[:, :])
