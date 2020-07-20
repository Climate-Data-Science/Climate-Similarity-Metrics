"""
    TODO: Module Docstring
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.basemap import Basemap
from scipy.stats import entropy
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

    In order to make the values of the different similarity measures comparable, they are binned in
    10%    bins using comparing.binning_values_to_quantiles.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List with similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales
                                           them in order to make the similarity values of different
                                           similarity measures comparable
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
        plot_similarities_whole_period(map_array, reference_series, measures,
                                       labels, scaling_func, level)
    elif mode == "whole_period_per_month":
        plot_similarities_whole_period_per_month(map_array, reference_series, measures,
                                                 labels, scaling_func, level)
    elif mode == "whole_period_winter_only":
        plot_similarities_winter_only(map_array, reference_series, measures,
                                      labels, scaling_func, level)
    else:
        print("Mode not available")


def plot_similarities_whole_period(map_array, reference_series, measures, labels,
                                   scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarity of a reference data series and all points on the map for the whole period
    regarding different similarity measures

    Each column contains a different similarity measure.

    In order to make the values of the different similarity measures comparable, they are binned
    in 10% bins using comparing.binning_values_to_quantiles.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        measures (list): List with similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales
                                           them in order to make the similarity values of different
                                           similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    axis = None
    fig, ax = plt.subplots(nrows=1, ncols=len(measures), figsize=(8*len(measures), 10))

    for i, measure in enumerate(measures):
        #Compute similarity
        sim_whole_period = calc.calculate_series_similarity(map_array,
                                                            reference_series,
                                                            level,
                                                            measure)
        #Check if only one map
        if (len(measures) == 1):
            axis = ax
        else:
            axis = ax[i]

        #Draw map
        plot_map(scaling_func(sim_whole_period), axis)
        axis.set_title(labels[i])

    fig.suptitle("Similarity between QBO and all other points for the whole period")
    plt.show()


def plot_similarities_whole_period_per_month(map_array, reference_series, measures, labels,
                                             scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarity of a reference data series and all points on the map for the whole period,
    but every month seperately, regarding different similarity measures

    Each column contains a different similarity measure and each row contains a different month.

    In order to make the values of the different similarity measures comparable, they are binned in
    10% bins using comparing.binning_values_to_quantiles.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        referenceSeries (numpy.ndarray): 1 dimensional reference series
        measures (list): List with similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales
                                           them in order to make the similarity values of different
                                           similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    axis = None
    len_measures = len(measures)
    fig, ax = plt.subplots(figsize=(8*len_measures, 14*len_measures), nrows=12, ncols=len(measures))

    for month in range(len(months)):
        #Check if only one map
        if (len_measures == 1):
            axis = ax[month]
        else:
            axis = ax[month][0]

        axis.set_ylabel(months[month])

        #Extract monthly values
        map_array_month = np.array([map_array[12 * i + month, :, :, :] for i in range(40)])
        reference_series_month = [reference_series[12 * i + month] for i in range(40)]

        for i, measure in enumerate(measures):
            #Calculate similarity
            similarity_month = calc.calculate_series_similarity(map_array_month,
                                                                reference_series_month,
                                                                level,
                                                                measure)
            if (len_measures == 1):
                axis = ax[month]
            else:
                axis = ax[month][i]

            #Plot Map
            scaled_similarity = scaling_func(similarity_month)
            plot_map(scaled_similarity, axis, colorbar=False)

    for i in range(len_measures):
        if (len_measures == 1):
            axis = ax[0]
        else:
            axis = ax[0][i]
        axis.set_title(labels[i])

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
    axis = None
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

        #Check if only one map
        if (len(measures) == 1):
            axis = ax
        else:
            axis = ax[i]

        #Draw map
        plot_map(scaling_func(sim_whole_period_winter), axis)

        axis.set_title(labels[i])

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
    axis = None
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
            if (len(measures) == 1):
                axis = ax
            else:
                axis = ax[i][j]
            axis.scatter(similarities[j], similarities[i])

    for i, label in enumerate(labels):
        if (len(measures) == 1):
            ax.set_ylabel(label)
            ax.set_title(label)
        else:
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
        combination_func (function): Function that combines two similarity values into one
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales
                                           them in order to make the similarity values of different
                                           similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    axis = None
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
            if (n_measures == 1):
                axis = ax
            else:
                axis = ax[i][j]
            combination = calc.combine_similarity_measures(similarities[i], similarities[j], combination_func)
            plot_map(combination[:], axis)

    for i, label in enumerate(labels):
        if (n_measures == 1):
            ax.set_ylabel(label)
            ax.set_title(label)
        else:
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
        combination_func (function): Function that combines two similarity values into one
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales
                                           them in order to make the similarity values of different
                                           similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    axis = None
    fig, ax = plt.subplots(nrows=1, ncols=len(measures), figsize=(8 * len(measures), 8))

    combination_func = comb.power_combination(combination_func)

    combinations = combinations_with_pearson(map_array, reference_series, combination_func, measures, labels,
                                             scaling_func, level)
    for i in range(len(combinations)):
        if (len(measures) == 1):
            axis = ax
        else:
            axis = ax[i]
        plot_map(combinations[i][:], axis)

    for i, label in enumerate(labels):
        if (len(measures) == 1):
            axis = ax
        else:
            axis = ax[i]
        axis.set_title(label)

    fig.suptitle("Combination with absolute values of Pearson's Correlation")
    plt.show()


def plot_sign_of_correlation_strength_of_both(map_array, reference_series, combination_func, measures, labels,
                                        scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the combinations of different similarity measures with Pearson's Correlation by taking
    the sign of Pearson's Correlation and combining the absolute values of Pearson's with the other
    similarity measure.

    The combination_func defines how the values are combined.

    Before the values are combined with the absolute values of Pearson's Correlation, they are scaled
    to make value ranges combinable (default: binned in 10% bins using comparing.binning_values_to_quantiles).

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        combination_func (function): Function that combines two similarity values into one
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    axis = None
    fig, ax = plt.subplots(nrows=1, ncols=len(measures), figsize=(8 * len(measures), 8))

    combination_func = comb.take_sign_first_strength_both(combination_func)

    combinations = combinations_with_pearson(map_array, reference_series, combination_func, measures, labels,
                                            scaling_func, level)
    for i in range(len(combinations)):
        if (len(measures) == 1):
            axis = ax
        else:
            axis = ax[i]
        plot_map(combinations[i][:], axis)

    for i, label in enumerate(labels):
        if (len(measures) == 1):
            axis = ax
        else:
            axis = ax[i]
        axis.set_title(label)

    fig.suptitle("Sign of Pearson's and values of both combined")
    plt.show()


def plot_level_of_agreement(map_array, reference_series, scoring_func, measures, labels,
                            scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot a map with the agreement of several similarity measures.
    For each similarity measure the scoring function will determine if there is a value that
    can be considered a dependency or not.

    The plotted map contains the percentages of how many of the similarity measures voted there is
    a dependency.

    Typical scoring function would be scoring_func = lambda x : x >= 0.8 and typical scaling function would
    be scaling_func=comp.binning_values_to_quantiles.
    Using this functions, the output map will show for how many similarity measures the similarity
    value between the time series of the point and the reference series is in the upper 20%.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        scoring_func (function): Function that takes in a value and outputs a boolean (whether there
                                 is a dependency or not)
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    #Compute agreement
    similarities = []
    agreement = np.zeros((256, 512))
    for i, measure in enumerate(measures):
        similarity = calc.calculate_series_similarity(map_array, reference_series, level, measure)
        similarities.append(scaling_func(similarity))
        n_measures = len(measures)

    for similarity_map in similarities:
        agreement = sum([agreement, np.vectorize(scoring_func)(similarity_map)])

    agreement = agreement / n_measures


    #Draw Map
    fig, (ax, cax) = plt.subplots(nrows=2,figsize=(12, 8),
                  gridspec_kw={"height_ratios":[1, 0.05]})
    plot_map(agreement, ax)

    #Draw Colorbar
    cmap = matplotlib.cm.viridis
    bounds = np.linspace(0, 100, n_measures + 2)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal',
                                ticks=np.linspace(0, 100, n_measures + 1),
                                boundaries=bounds)

    plt.title("Level of agreement (in %) between {}".format(labels))
    plt.show()


def plot_std_between_similarity_measures(map_array, reference_series, measures, labels, threshold=0.35,
                            scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot a map with the standard deviation between all similarity values, using a list of similarity measures,
    between the reference series and the time series for each point.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        threshold (float64, optional): Percentage-Threshold to decide when a value is high, low, or between
            Defaults to 0.35 (35%)
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    #Compute Standard Deviation
    title = ["Agree high values (Top {}%)".format(threshold*100), "Agree low values (Lowest {}%)".format(threshold*100), "Not sure"]
    similarities = []
    maps = []
    agreement = np.zeros((256, 512))
    high_map = np.ones((256, 512))
    low_map = np.ones ((256, 512))
    between = np.ones((256, 512))
    for i, measure in enumerate(measures):
        similarity = scaling_func(calc.calculate_series_similarity(map_array, reference_series,
                                                                  level, measure))
        similarities.append(similarity)
        high_map = high_map * (similarity >= (1 - threshold))
        low_map = low_map * (similarity <= threshold)

    between = between * (1 - high_map)
    between = between * (1 - low_map)

    agreement = np.std(similarities, axis=0)

    maps = [high_map, low_map, between]

    cmap = plt.cm.get_cmap("viridis").reversed()

    #Draw Maps
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(32,14))

    for i, map in enumerate(maps):
        masked_agreement = np.ma.array(agreement, mask=(map == 0))
        plot_map(masked_agreement, ax[i], cmap=cmap, invert_colorbar=True)
        ax[i].set_title(title[i])

    fig.suptitle("Agreeableness defined with standard deviation between \n {}".format(labels))
    plt.show()


def plot_entropy_between_similarity_measures(map_array, reference_series, measures, labels, threshold=0.35,
                            scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot a map with the entropy between all similarity values, using a list of similarity measures, between
    the reference series and the time series for each point.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        threshold (float64, optional): Percentage-Threshold to decide when a value is high, low, or between
            Defaults to 0.35 (35%)
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    #Compute Entropy
    title = ["Agree high values (Top {}%)".format(threshold*100), "Agree low values (Lowest {}%)".format(threshold*100), "Not sure"]
    similarities = []
    maps = []
    agreement = np.zeros((256, 512))
    high_map = np.ones((256, 512))
    low_map = np.ones ((256, 512))
    between = np.ones((256, 512))

    for i, measure in enumerate(measures):
        similarity = scaling_func(calc.calculate_series_similarity(map_array, reference_series,
                                                                  level, measure))
        similarities.append(similarity)
        high_map = high_map * (similarity >= (1 - threshold))
        low_map = low_map * (similarity <= threshold)

    between = between * (1 - high_map)
    between = between * (1 - low_map)

    agreement = entropy(similarities, axis=0)

    maps = [high_map, low_map, between]

    cmap = plt.cm.get_cmap("viridis")

    #Draw Maps
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(32,14))

    for i, map in enumerate(maps):
        masked_agreement = np.ma.array(agreement, mask=(map == 0))
        plot_map(masked_agreement, ax[i], cmap=cmap, invert_colorbar=True)
        ax[i].set_title(title[i])

    fig.suptitle("Agreeableness defined with entropy between \n {}".format(labels))
    plt.show()


def combinations_with_pearson(map_array, reference_series, combination_func, measures, labels,
                                        scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Return the combinations of values from different similarity measures with the of
    Pearson's Correlation. The combination_func defines how the values are combined.

    Before the values are combined with the absolute values of Pearson's Correlation, they are scaled
    to make value ranges combinable (default: binned in 10% bins using comparing.binning_values_to_quantiles).

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        combination_func (function): Function that combines two similarity values into one
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0

    Returns:
        Array with the resulting combination arrays
    """
    similarities = []
    combinations = []
    for i, measure in enumerate(measures):
        similarity = calc.calculate_series_similarity(map_array, reference_series, level, measure)
        similarities.append(scaling_func(similarity))
    n_measures = len(measures)

    pearson_similarity = calc.calculate_series_similarity(map_array, reference_series, level, sim.pearson_correlation)

    for i in range(n_measures):
        combination = calc.combine_similarity_measures(pearson_similarity, similarities[i], combination_func)
        combinations.append(combination)

    return combinations


def plot_time_delayed_dependencies(map_array, reference_series, time_shifts, measures, labels,
                                        scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarities for different similarity measures between a reference series and the map at different time steps.

    Before computing the similarity, the map is shifted by a given index and the reference series stays unchanged.

    The results are made comparable using the scaling_func. The results of Pearson's Correlation stay unscaled.


    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        time_shifts (array): List of integers that indicate by how many time units the reference series should be shifted
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    axis = None
    #Compute time delayed similarities
    len_time = map_array.shape[0]
    len_time_shifts = len(time_shifts)
    len_measures = len(measures)
    fig, ax = plt.subplots(nrows=len_time_shifts, ncols=len_measures, figsize=(10 * len_measures, 14 * len_time_shifts))

    for j, shift in enumerate(time_shifts):
        shifted_reference_series = calc.shift(reference_series, shift)
        for i, measure in enumerate(measures):
            similarity = calc.calculate_series_similarity(map_array, shifted_reference_series, level, measure)

            #Scale results for similarity measures different than Pearson's
            if (measure != sim.pearson_correlation or measure !=sim.pearson_correlation_abs):
                similarity = scaling_func(similarity)

            #Exceptional cases
            if len_measures == 1:
                if len_time_shifts == 1:
                    axis = ax
                else:
                    axis = ax[j]
            else:
                if len_time_shifts == 1:
                    axis = ax[i]
                else:
                    axis = ax[j][i]

            #Plot results on map
            plot_map(similarity, axis)

    for i in range(len_measures):
        if len_measures == 1:
            if len_time_shifts == 1:
                axis = ax
            else:
                axis = ax[0]
        else:
            if len_time_shifts == 1:
                axis = ax[i]
            else:
                axis = ax[0][i]
        axis.set_title(labels[i])

    for j in range(len_time_shifts):
        if len_measures == 1:
            if len_time_shifts == 1:
                axis = ax
            else:
                axis = ax[j]
        else:
            if len_time_shifts == 1:
                axis = ax[0]
            else:
                axis = ax[j][0]
        axis.set_ylabel("Shifted by {}".format(time_shifts[j]))

    fig.suptitle("Similarities to different time steps")


def plot_map(values, axis, cmap=plt.cm.get_cmap("viridis"), colorbar=True, invert_colorbar=False):
    """
    Plot values on a Basemap map

    Args:
        values (numpy.ndarray): 2-d array with dimensions latitude and longitude
        axis: Axis on which the map should be displayed
        cmap (optional): matplotlib.Colormap to use
            Defaults to plt.cm.get_cmap("viridis")
        colorbar (boolean, optional): Boolean indicating if a colorbar should be plotted
            Defaults to True
        invert_colorbar (boolean, optional): Boolean indicating if the colobar should be inverted
    """
    #Create map
    m = Basemap(projection='mill', lon_0=30, resolution='l', ax=axis)
    m.drawcoastlines()
    lons, lats = m.makegrid(512, 256)
    x, y = m(lons, lats)

    #Draw values in map
    cs = m.contourf(x, y, values, cmap=cmap)
    if colorbar:
        cbar = m.colorbar(cs, location='bottom', pad="5%")
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
        if invert_colorbar:
            cbar.ax.invert_xaxis()
