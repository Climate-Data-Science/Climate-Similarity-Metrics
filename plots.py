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
    fig, ax = plt.subplots(nrows=1, ncols=len(measures), figsize=(14*len(measures), 10))

    for i, measure in enumerate(measures):
        #Compute similarity
        sim_whole_period = calc.calculate_series_similarity(map_array,
                                                            reference_series,
                                                            level,
                                                            measure)
        #Check if only one map
        axis = check_axis(ax, column=i, column_count=len(measures))

        #Draw map
        plot_map(scaling_func(sim_whole_period), axis)

    annotate(ax, column_count=len(measures), column_labels=labels)
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
    len_measures = len(measures)
    fig, ax = plt.subplots(figsize=(14*len_measures, 10*len_measures), nrows=12, ncols=len(measures))

    for month in range(len(months)):
        #Extract monthly values
        map_array_month = np.array([map_array[12 * i + month, :, :, :] for i in range(40)])
        reference_series_month = [reference_series[12 * i + month] for i in range(40)]

        for i, measure in enumerate(measures):
            #Calculate similarity
            similarity_month = calc.calculate_series_similarity(map_array_month,
                                                                reference_series_month,
                                                                level,
                                                                measure)
            axis = check_axis(ax, row=month, column=i, row_count=len(months), column_count=len_measures)

            #Plot Map
            scaled_similarity = scaling_func(similarity_month)
            plot_map(scaled_similarity, axis, colorbar=False)

    annotate(ax, row_count=len(months), column_count=len_measures, row_labels=months, column_labels=labels)
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
    fig, ax = plt.subplots(nrows=1, ncols=len(measures), figsize=(14*len(measures), 10))

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
        axis = check_axis(ax, column=i, column_count=len(measures))

        #Draw map
        plot_map(scaling_func(sim_whole_period_winter), axis)

    annotate(ax, column_count=len(measures), column_labels=labels)
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
    fig, ax = plt.subplots(nrows=n_measures, ncols=n_measures, figsize=(14 * n_measures, 10 * n_measures))

    for i, measure_i in enumerate(measures):
        for j, measure_j in enumerate(measures):
            axis = check_axis(ax, row=i, column=j, row_count=n_measures, column_count=n_measures)
            axis.scatter(similarities[j], similarities[i])

    annotate(ax, row_count=n_measures, column_count=n_measures, row_labels=labels, column_labels=labels)
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
        combination_func (function): Function that combines a list of similarity maps into one
        measures (list): List of similarity measures to compute similarity between two time series
        labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales
                                           them in order to make the similarity values of different
                                           similarity measures comparable
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))

    combination = combination_func(similarities)
    plot_map(combination, ax)

    fig.suptitle("Combination of {} by {}".format(labels, combination_func.__name__))
    plt.show()

def plot_sign_of_correlation_strength_of_all(map_array, reference_series, combination_func, measures, labels,
                                        scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the combination of different similarity measures by taking
    the sign of Pearson's Correlation and combining the absolute values of the similarity measures using combination_func

    The combination_func defines how the values are combined.

    Before the values are combined, they are scaled to make value ranges combinable
    (default: binned in 10% bins using comparing.binning_values_to_quantiles).

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
    #Compute similarities
    sign_map = None
    similarities = []
    for i, measure in enumerate(measures):
        similarity = calc.calculate_series_similarity(map_array, reference_series, level, measure)
        if measure == sim.pearson_correlation:
            sign_map = similarity
        similarities.append(scaling_func(similarity))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14 * len(measures), 10))

    combination = comb.combine_power_with_sign(combination_func, similarities, sign_map)

    plot_map(combination, ax)

    fig.suptitle("Sign of Pearson's and values of {} combined by {}".format(labels, combination_func.__name__))
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
    fig, (ax, cax) = plt.subplots(nrows=2,figsize=(14, 10),
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


def plot_agreement_areas_defined_with(map_array, reference_series, measures, measure_labels, agreement_func,
                                value_thresholds, agreement_thresholds, filter_values_high=True, filter_agreement_high=False,
                                scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot areas where the similarity measures agree on the dependencies.
    Contains the following steps:
        1. Compute similarity between reference series and map with every similarity measure
        2. Combine the similarity maps into two summary maps:
            - Combine using np.mean to get a summary value for the similarity measures
            - Combine using agreement_func to get an agreement value for the similarity measures
        3. Filter the maps using their respective thresholds
        4. Plot map containing ones(point has satisfied both conditions) and zeros(not satisfied at least one condition).
        5. Repeat 3-4 for every combination of value thresholds and agreement thresholds

    Before the values are combined (Step 2), they are scaled with the scaling_func to make value ranges combinable.
    Pearson's Correlation will not be scaled.

    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List of similarity measures to compute similarity between two time series
        measure_labels (List): Labels for the similarity measures
        agreement_func (function): Function to compute agreement between similarity values
        value_thresholds (List): List of thresholds to filter the combined similarity values on
        agreement_thresholds (List): List of thresholds to filter the agreement on
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
    """
    n_vt = len(value_thresholds)
    n_at = len(agreement_thresholds)
    maps = calc.calculate_filtered_agreement_areas_threshold_combinations(map_array, reference_series, measures, value_thresholds,
                                                                          agreement_thresholds, agreement_func=agreement_func,
                                                                          filter_values_high=filter_values_high,
                                                                          filter_agreement_high=filter_agreement_high,
                                                                          scaling_func=scaling_func, level=level)
    fig, ax = plt.subplots(nrows=n_vt, ncols=n_at, figsize=(14*n_at, 10*n_vt))
    for i, value_threshold in enumerate(value_thresholds):
        for j, agreement_threshold in enumerate(agreement_thresholds):
            axis = check_axis(ax, row=i, column=j, row_count=n_vt, column_count=n_at)
            plot_map(maps[i, j, :, :], axis, colorbar=False, cmap=plt.cm.get_cmap("Blues"))

    row_labels = ["Value Threshold of {}".format(str(i)) for i in value_thresholds]
    column_labels = ["Agreement Threshold of {}".format(str(j)) for j in agreement_thresholds]
    annotate(ax, row_count=n_vt, column_count=n_at, row_labels=row_labels, column_labels=column_labels)

    fig.suptitle("Agreement areas between {} defined with {}".format(measure_labels, agreement_func.__name__))



def plot_time_delayed_dependencies(map_array, reference_series, time_shifts, measures, measure_labels,
                                        scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarities for different similarity measures between a reference series and the map delayed by different time steps.

    Before computing the similarity, the map is shifted by a given index and the reference series stays unchanged.

    The results are made comparable using the scaling_func. The results of Pearson's Correlation stay unscaled.


    Args:
        map_array (numpy.ndarray): Map with 4 dimensions - time, level, latitude, longitude
        reference_series (numpy.ndarray): 1 dimensional reference series
        time_shifts (array): List of integers that indicate by how many time units the map should be shifted
        measures (list): List of similarity measures to compute similarity between two time series
        measure_labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    #Compute time delayed similarities
    len_time_shifts = len(time_shifts)
    len_measures = len(measures)
    shifted_map = map_array
    short_reference_series = reference_series
    fig, ax = plt.subplots(nrows=len_time_shifts, ncols=len_measures, figsize=(14 * len_measures, 10 * len_time_shifts))

    for j, shift in enumerate(time_shifts):
        if shift > 0:
            short_reference_series = reference_series[:-shift]
            shifted_map = map_array[shift:, :,:,:]
        for i, measure in enumerate(measures):
            similarity = calc.calculate_series_similarity(shifted_map, short_reference_series, level, measure)

            #Scale results for similarity measures different than Pearson's
            if ((measure != sim.pearson_correlation) & (measure != sim.pearson_correlation_abs)):
                similarity = scaling_func(similarity)

            #Check axis
            axis = check_axis(ax, row=j, column=i, row_count=len_time_shifts, column_count=len_measures)

            #Plot results on map
            plot_map(similarity, axis)

    #Annotate rows and columns
    shift_labels = ["Shifted by {}".format(i) for i in time_shifts]
    annotate(ax, row_count=len_time_shifts, column_count=len_measures, row_labels=shift_labels, column_labels=measure_labels)
    fig.suptitle("Similarities to different time steps")

def plot_similarities_to_different_datasets(datasets, dataset_labels, reference_series, measures, measure_labels,
                                        scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarities for different similarity measures between a reference series and different datasets.

    The results are made comparable using the scaling_func. The results of Pearson's Correlation stay unscaled.


    Args:
        datasets (list): List with datasets to compute the similarity to
        dataset_labels (list): List of labels for the datasets
        reference_series (numpy.ndarray): 1 dimensional reference series
        measures (list): List of similarity measures to compute similarity between two time series
        measure_labels (list): List of labels for the measures
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    n_datasets = len(datasets)
    len_measures = len(measures)
    fig, ax = plt.subplots(nrows=n_datasets, ncols=len_measures, figsize=(14 * len_measures, 10 * n_datasets))

    for j, file in enumerate(datasets):
        for i, measure in enumerate(measures):
            similarity = calc.calculate_series_similarity(file, reference_series, level, measure)

            #Scale results for similarity measures different than Pearson's
            if ((measure != sim.pearson_correlation) & (measure != sim.pearson_correlation_abs)):
                similarity = scaling_func(similarity)

            #Check axis
            axis = check_axis(ax, row=j, column=i, row_count=n_datasets, column_count=len_measures)

            #Plot results on map
            plot_map(similarity, axis)

    #Annotate rows and columns
    annotate(ax, row_count=n_datasets, column_count=len_measures, row_labels=dataset_labels, column_labels=measure_labels)
    fig.suptitle("Similarities to different datasets")


def plot_time_delayed_similarities_to_different_datasets(datasets, dataset_labels, reference_series, time_shifts, measure,
                                                         scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the similarities between a reference series and different datasets delayed by different time steps.

    Before computing the similarity, the dataset is shifted by a given index and the reference series stays unchanged.
    This procedure is repeated for every index-shit (time_shifts) and for every dataset.

    The results are made comparable using the scaling_func. The results of Pearson's Correlation stay unscaled.


    Args:
        datasets (list): List with datasets to compute the similarity to
        dataset_labels (list): List of labels for the datasets
        reference_series (numpy.ndarray): 1 dimensional reference series
        time_shifts (array): List of integers that indicate by how many time units the dataset should be shifted
        measure (function): Similarity measure to compute similarity between two time series
        scaling_func (function, optional): Function that takes a map of similarity values and scales them in order
                                           to make the similarity values of different similarity measures comparable
            Defaults to comp.binning_values_to_quantiles
        level (int, optional): Level on which the similarity should be calculated
            Defaults to 0
    """
    n_datasets = len(datasets)
    len_shifts = len(time_shifts)
    short_reference_series = reference_series
    fig, ax = plt.subplots(nrows=n_datasets, ncols=len_shifts, figsize=(14 * len_shifts, 10 * n_datasets))

    for i, shift in enumerate(time_shifts):
        for j, dataset in enumerate(datasets):
            shifted_dataset = dataset
            if shift > 0:
                short_reference_series = reference_series[:-shift]
                shifted_dataset = dataset[shift:, :,:,:]
            similarity = calc.calculate_series_similarity(shifted_dataset, short_reference_series, level, measure)

            #Scale results for similarity measures different than Pearson's
            if ((measure != sim.pearson_correlation) & (measure != sim.pearson_correlation_abs)):
                similarity = scaling_func(similarity)

            #Check axis
            axis = check_axis(ax, row=j, column=i, row_count=n_datasets, column_count=len_shifts)

            #Plot results on map
            plot_map(similarity, axis)

    #Annotate rows and columns
    shift_labels = ["Shifted by {}".format(i) for i in time_shifts]
    annotate(ax, row_count=n_datasets, column_count=len_shifts, row_labels=dataset_labels, column_labels=shift_labels)

    fig.suptitle("Similarities to different datasets for different time delays using {}".format(measure.__name__))


def plot_time_delayed_agreeableness_to_different_datasets(datasets, dataset_labels, reference_series, time_shifts, measures,
                                                          measure_labels, value_threshold, agreement_threshold, agreement_func = np.std,
                                                          filter_values_high=True, filter_agreement_high=False,
                                                          scaling_func=comp.binning_values_to_quantiles, level=0):
    """
    Plot the areas where the similarity values agree on dependency between a reference series and different datasets delayed by different
    time steps.

    Before computing the similarity, the dataset is shifted by a given index and the reference series stays unchanged.
    This procedure is repeated for every index-shit (time_shifts) and for every dataset and for every similarity measure.
    Then a degree of agreement areas between the similarity values is calculated using calc.calculate_filtered_agreement_areas.

    Args:
        datasets (list): List with datasets to compute the similarity to
        dataset_labels (list): List of labels for the datasets
        reference_series (numpy.ndarray): 1 dimensional reference series
        time_shifts (array): List of integers that indicate by how many time units the dataset should be shifted
        measures (list): List of similarity measures to compute similarity between two time series
        measure_labels (list): List of labels for the similarity measures
        value_threshold (float): Threshold to filter the combined similarity values on
        agreement_threshold (float): Threshold to filter the agreement between the similarity values on
        agreement_func (function, optional): Agreeableness measure to compute degree of agreement between
                                                    similarity values
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
    """
    n_datasets = len(datasets)
    len_shifts = len(time_shifts)
    short_reference_series = reference_series
    fig, ax = plt.subplots(nrows=n_datasets, ncols=len_shifts, figsize=(14 * len_shifts, 10 * n_datasets))

    for i, shift in enumerate(time_shifts):
        for j, dataset in enumerate(datasets):
            shifted_dataset = dataset
            if shift > 0:
                short_reference_series = reference_series[:-shift]
                shifted_dataset = dataset[shift:, :,:,:]
            map = calc.calculate_filtered_agreement_areas(shifted_dataset, short_reference_series, measures, value_threshold, agreement_threshold,
                                                          agreement_func=agreement_func, filter_values_high=filter_values_high,
                                                          filter_agreement_high=filter_agreement_high,scaling_func=scaling_func, level=level)
            axis = check_axis(ax, row=j, column=i, row_count=n_datasets, column_count=len_shifts)
            plot_map(map, axis, colorbar=False, cmap=plt.cm.get_cmap("Blues"))

    #Annotate rows and columns
    shift_labels = ["Shifted by {}".format(i) for i in time_shifts]
    annotate(ax, row_count=n_datasets, column_count=len_shifts, row_labels=dataset_labels, column_labels=shift_labels)

    fig.suptitle("Agreeableness between {} to different datasets for different time delays using {} (Value threshold: {}, Agreement threshold: {})"
       .format(measure_labels, agreement_func.__name__, value_threshold, agreement_threshold))


def plot_map(values, axis, cmap=plt.cm.get_cmap("viridis"), colorbar=True, invert_colorbar=False,
             overwrite_colorbar_boundaries=False, colorbar_min=0, colorbar_max=1):
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
        overwrite_colorbar_boundaries (boolean, optional): Boolean indicating if the colorbar
                                                           boundaries should be overwritten
            Defaults to False
        colorbar_min (int, optional): Minimum value for colorbar (if colorbar is overwritten)
            Defaults to 0
        colorbar_max (int, optional): Maximum value for colorbar (if colorbar is overwritte)
            Defaults to 1
    """
    #Create Colormap
    vmin = np.min(values)
    vmax = np.max(values)
    if overwrite_colorbar_boundaries:
        vmin = colorbar_min
        vmax = colorbar_max
    #Create map
    m = Basemap(projection='mill', lon_0=30, resolution='l', ax=axis)
    m.drawcoastlines()
    lons, lats = m.makegrid(512, 256)
    x, y = m(lons, lats)

    #Draw values in map
    cs = m.contourf(x, y, values, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        #Create Colorbar
        cbar = m.colorbar(cs, location='bottom', pad="5%")
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
        if invert_colorbar:
            cbar.ax.invert_xaxis()

def check_axis(ax, row=0, column=0, row_count=1, column_count=1):
    axis = None
    if row_count == 1:
        if column_count == 1:
            axis = ax
        else:
            axis = ax[column]
    else:
        if column_count == 1:
            axis = ax[row]
        else:
            axis = ax[row][column]
    return axis


def annotate(ax, row_count=1, column_count=1, row_labels=None, column_labels=None):
    if (column_count > 0) & (column_labels != None):
        for i in range(column_count):
            axis = check_axis(ax, row=0, column=i, row_count=row_count, column_count=column_count)
            axis.set_title(column_labels[i])

    if (row_count > 0) & (row_labels != None):
        for j in range(row_count):
            axis = check_axis(ax, row=j, column=0, row_count=row_count, column_count=column_count)
            axis.set_ylabel(row_labels[j])
