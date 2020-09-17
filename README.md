# Climate-Similarity-Metrics
Which similarity metrics are the most helpful to understand climate?

## Table of Contents

**Randomized Dependence Coefficient** - Contains the implementation of the 'randomized dependence coefficient' similarity measure

**Similarity Measures** - Contains a summary of all the time series similarity measures found in literature

**Writing in Science** - Contains summaries for the 'Writing in Science'-Course on Coursera

**data** - Contains the data used in this work

**papers** - Contains summaries of related work

**0_vizualize.ipynb** - Visualizes the data using the matplotlib basemap toolkit

**10_agreeableness_defined_with_std (Widget version).ipynb** - Puts at disposal a widget to try different value- and agreement-threshold combinations

**10_agreeableness_defined_with_std_and_entropy.ipynb** - Presents the notion of agreeableness between similarity measures

**11_level_of_agreement.ipynb** - Presents the initial idea with level of agreement

**12_time_delayed_dependencies.ipynb** - Computes and plots the dependencies delayed in time

**13_dependencies_to_different_levels.ipynb** - Computes and plots the dependencies over different pressure levels

**14_time_delayed_dependencies_and_agreeableness_to_different_levels.ipynb** - Computes and plots the dependencies and the agreeableness between different similarity measures over levels and delayed in time

**15_applications.ipynb** - Shows some further applications for agreeableness

**1_plotting_1979-2019.ipynb** - Plots the mean per longitude of the u wind component for every latitude of every year

**2_point-wise_similarities.ipynb** - Computes the similarity to a random point for every point on the map

**3_deriving_QBO.ipynb** - Derives different QBO indices from the data

**4_converting_coordinates_to grid_indices.ipynb** - Converts geographical coordinates into the corresponding indices for the gaussian grid

**5_similarity_to_qbo.ipynb** - Computes the Mutual Information and Pearson Correlation between the QBO and the whole map at 30 hPa (for the whole period, for the whole period but every month seperately and for the whole period but only winter months are taken into account)

**6_different_similarity_measures.ipynb** - Presents the different similarity measures implemented in the similarity_measures.py file

**7_compare_similarity_measures.ipynb** - Presents different functions to make similarity measures comparable

**8_combine_similarity_measures.ipynb** - Presents different functions to combine the similarity measures


**9_dependencies_between_similarity_measures.ipynb** - Visualizes the relationship between different similarity measures

**bachelor_thesis_figures.ipynb** - Contains all the figures used in the thesis

**calculations.py** - Is the module for the similarity and agreeableness computation between time series similarity measures.

**combining.py** - Is the module containing different functions to combine results of similarity measures

**comparing.py** - Is the module containing different functions to make results of similarity measures comparable

**environment.yaml** - Contains an anaconda environment with all the required packages

**plots.py** - Is the module for visualizations of the similarity and agreeableness computation results.

**proposal_figures.ipynb** - Contains the figures used in the proposal presentation

**similarity_measures.py** - Is the module containing different similarity measures for time series

## Goal of the Project
Create a modular framework for analysing and understanding relationships in climate data. The moduls should allow for testing different climate indices, similarity metrics, and time-scales. The results should be interpretable by climate scientists.

## Data Description

The the *u* component (east-west) of the wind for 512 longitudes, 256 latitudes, and 37 altitudes aggregated by months.

## Data Setup

1. Download `era-int_pl_1981-mm-u.nc` file from this [repository](https://nextcloud.scc.kit.edu/s/cwpp3wdQPcm96jq).
2. Place the data file `era-int_pl_1981-mm-u.nc` in the directory `data/`.
3. Use `era-int_pl_1979-2019-mm-u.nc` file from this [repository](https://nextcloud.scc.kit.edu/s/cwpp3wdQPcm96jq) for the "Data Preprocessing" step

##  Data Preprocessing
Use the data manipulation tool cdo to extract all values for pressure level 3, 30, 70 and 300 hPa.
Example for extracting values for pressure level 30 with the following command:

`cdo -select,level=30 era-int_pl_1979-2019-mm-u.nc era-int_pl_1979-2019-mm-l30-u.nc`

Analogously for 3, 70 and 300

## Environment Setup

1. Create a new conda environment with all the required dependencies:

`conda env create -f environment.yaml`

2. Activate the environment:

`conda activate climate_similarity_measures`

3. Navigate to the `Randomized Dependence Coefficient` folder and run `python setup.py install`

## QBO Index

### 1st Possibility

Values at Singapore (1N, 104E) at pressure level 30 hPa

### 2nd Possibility

Values at +- 5°north and south of the equator at pressure level 30 hPa

### Data from the Internet to cross check

Data to cross check can be found under the following link:

https://acd-ext.gsfc.nasa.gov/Data_services/met/qbo/QBO_Singapore_Uvals_GSFC.txt

The file has to be placed in the directory `data/`

You have to **remove the first 9 lines** in order to make it readable for Python.

If you want to skip this step, you can download a preprocessed version [here](https://1drv.ms/t/s!AjmYENXxse7Bgu5zq1jKa4INqs6MoQ?e=qbyuof)

## Useful Materials
* [How to Write a Good Git Commit Message](https://chris.beams.io/posts/git-commit/)
* [Python Styleguide by Google](http://google.github.io/styleguide/pyguide.html)
