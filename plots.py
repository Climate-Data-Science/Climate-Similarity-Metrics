"""
    TODO: Module Docstring
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.basemap import Basemap
import calculations as calc
import comparing as comp
import combining as comb
import similarity_measures as sim

months = ["January", "February", "March", "April", "May",
          "June", "July", "August", "September", "October",
          "November", "December"]

def plot_similarities(map_array, reference_series, measures, labels,
                      scaling_func=comp.binning_values_to_quantiles, level=0, mode="whole_period"):
    """
    Plot the similarity of a reference data series and all points on the map regarding different
    similarity measures.

    In order to make the values of the different similarity measures comparable, they are binned in 10%
    bins using comparing.binning_values_to_quantiles.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List with similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
        mode (str, optional): Mode of visualization
            Options: "whole_period": Similarity over whole period
                     "whole_period_per_month": Similarity over whole period, every month seperately
                     "whole_period_winter_only": Similarity over whole period, only winter months
            Defaults to "whole_period"
    """
    if mode == "whole_period":
        plot_similarities_whole_period(map_array, reference_series, measures, labels, scaling_func, level)
    elif mode == "whole_period_per_month":
        plot_similarities_whole_period_per_month(map_array, reference_series, measures, labels, scaling_func, level)
    elif mode == "whole_period_winter_only":
        plot_similarities_winter_only(map_array, reference_series, measures, labels, scaling_func, level)
    else:
        print("Mode not available")


def plot_similarities_whole_period(map_array, reference_series, measures, labels,
                                   scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarity of a reference data series and all points on the map for the whole period
    regarding different similarity measures

    Each column contains a different similarity measure.

    In order to make the values of the different similarity measures comparable, they are binned in 10%
    bins using comparing.binning_values_to_quantiles.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        measures (list): List with similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """

    fig, ax = plt.subplots(nrows=1, ncols=len(measures), figsize=(8*len(measures), 10))

    for i, measure in enumerate(measures):
        #Compute similarity
        sim_whole_period = calc.calculate_series_similarity(map_array,
                                                            reference_series,
                                                            level,
                                                            measure)
        #Draw map
        plot_map(scaling_func(sim_whole_period), ax[i])
        ax[i].set_title(labels[i])

    fig.suptitle("Similarity between QBO and all other points for the whole period")
    plt.show()


def plot_similarities_whole_period_per_month(map_array, reference_series, measures, labels,
                                             scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarity of a reference data series and all points on the map for the whole period,
    but every month seperately, regarding different similarity measures

    Each column contains a different similarity measure and each row contains a different month.

    In order to make the values of the different similarity measures comparable, they are binned in 10%
    bins using comparing.binning_values_to_quantiles.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        measures (list): List with similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    fig, ax = plt.subplots(figsize=(8*len(measures), 14*len(measures)), nrows=12, ncols=len(measures))

    for month in range(len(months)):
        ax[month][0].set_ylabel(months[month])

        #Extract monthly values
        map_array_month = np.array([map_array[12 * i + month, :, :, :] for i in range(40)])
        reference_series_month = [reference_series[12 * i + month] for i in range(40)]

        for i, measure in enumerate(measures):
            ax[0][i].set_title(labels[i])

            #Calculate similarities
            similarity_month = calc.calculate_series_similarity(map_array_month,
                                                                reference_series_month,
                                                                level,
                                                                measure)

            #Plot Map
            m = Basemap(projection='mill', lon_0=30, resolution='l', ax=ax[month][i])
            m.drawcoastlines()
            lons, lats = m.makegrid(512, 256)
            x, y = m(lons, lats)
            cs = m.contourf(x, y, scaling_func(similarity_month))

    fig.suptitle("Similarity between QBO and all other points 1979 - 2019 per month")
    plt.show()


def plot_similarities_winter_only(map_array, reference_series, measures, labels,
                                  scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarity of a reference data series and all points on the map for the whole
    period, but only winter months are taken into account, regarding different similarity
    measures

    Each column contains a different similarity measure.

    In order to make the values of the different similarity measures comparable, they are binned in 10%
    bins using comparing.binning_values_to_quantiles.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        measures (list): List with similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """

    fig, ax = plt.subplots(nrows=1, ncols=len(measures), figsize=(8*len(measures), 10))

    winter_indices = []
    for i in range(40):
        year = 12 * i
        winter_indices.append(year) #January
        winter_indices.append(year + 1) #February
        winter_indices.append(year + 11) #December

    #Extract winter values
    reference_series_winter = reference_series[winter_indices]
    map_array_winter = map_array[winter_indices, :, :, :]

    for i, measure in enumerate(measures):
        #Compute similarity
        sim_whole_period_winter = calc.calculate_series_similarity(map_array_winter,
                                                                   reference_series_winter,
                                                                   level,
                                                                   measure)
        #Draw map
        plot_map(scaling_func(sim_whole_period_winter), ax[i])

        ax[i].set_title(labels[i])

    fig.suptitle("Similarity between QBO and all other points 1979 - 2019 for Winter months")
    plt.show()


def plot_similarity_dependency(map_array, reference_series, measures, labels, level=0):
    """
    Plot a matrix of dependcies between two similarity measures with one similarity
    measure on the x-axis and one on the y-axis

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    #Compute similarities
    similarities = []
    for i, measure in enumerate(measures):
        similarities.append(np.array(calc.calculate_series_similarity(map_array,
                                                                      reference_series,
                                                                      level,
                                                                      measure)))

    n_measures = len(measures)
    #Plot dependencies in matrix
    fig, ax = plt.subplots(nrows=n_measures, ncols=n_measures, figsize=(8 * n_measures, 8 * n_measures))

    for i, measure_i in enumerate(measures):
        for j, measure_j in enumerate(measures):
            ax[i][j].scatter(similarities[j], similarities[i])

    for i, label in enumerate(labels):
        ax[i][0].set_ylabel(label)
        ax[0][i].set_title(label)

    fig.suptitle("Dependency between pairs of similarity measures")
    plt.show()


def plot_similarity_measures_combinations(map_array, reference_series, combination_func, measures, labels,
                                         scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot a matrix of combinations of two similarity measures. The combination_func defines how the
    values are combined.

    Before the values are combined, they are binned in 10% bins using
    comparing.binning_values_to_quantiles.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        combination_func (function): Function that comines two similarity values into one
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    #Compute similarities
    similarities = []
    for i, measure in enumerate(measures):
        sim = calc.calculate_series_similarity(map_array, reference_series, level, measure)
        similarities.append(scaling_func(sim))

    n_measures = len(measures)
    #Plot dependencies in matrix
    fig, ax = plt.subplots(nrows=n_measures, ncols=n_measures, figsize=(8 * n_measures, 8 * n_measures))


    for i in range(n_measures):
        for j in range(n_measures):
            combination = calc.combine_similarity_measures(similarities[i], similarities[j], combination_func)
            plot_map(combination[:], ax[i][j])

    for i, label in enumerate(labels):
        ax[i][0].set_ylabel(label)
        ax[0][i].set_title(label)

    fig.suptitle("Combination of similarity measures")
    plt.show()


def plot_power_of_dependency(map_array, reference_series, combination_func, measures, labels,
                             scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the combinations of values from different similarity measures with the absolute values of
    Pearson's Correlation. Taking the absolute value of Pearson's will eliminate the direction of
    the dependency and only the information about the strength of dependency will remain.

    The combination_func defines how the values are combined.

    Before the values are combined with the absolute values of Pearson's Correlation, they are scaled
    to make value ranges combinable (default: binned in 10% bins using comparing.binning_values_to_quantiles).

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        combination_func (function): Function that comines two similarity values into one
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    similarities = []
    for i, measure in enumerate(measures):
        similarity = calc.calculate_series_similarity(map_array, reference_series, level, measure)
        similarities.append(scaling_func(similarity))
    n_measures = len(measures)

    pearson_similarity = calc.calculate_series_similarity(map_array, reference_series, level, sim.pearson_correlation)

    fig, ax = plt.subplots(nrows=1, ncols=n_measures, figsize=(8 * n_measures, 8))

    combination_func = comb.power_combination(combination_func)

    for i in range(n_measures):
        combination = calc.combine_similarity_measures(pearson_similarity, similarities[i], combination_func)
        plot_map(combination[:], ax[i])

    for i, label in enumerate(labels):
        ax[i].set_title(label)

    fig.suptitle("Combination with absolute values of Pearson's Correlation")
    plt.show()


def plot_map(values, axis):
    """
    Plot values on a Basemap maps

    Args:
        values (numpy.ndarray): 2-d array with dimensions latitude and longitude
        axis: Axis on which the map should be displayed
    """
    m = Basemap(projection='mill', lon_0=30, resolution='l', ax=axis)
    m.drawcoastlines()
    lons, lats = m.makegrid(512, 256)
    x, y = m(lons, lats)

    #Draw similarity
    cs = m.contourf(x, y, values)
    cbar = m.colorbar(cs, location='bottom', pad="5%")
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
