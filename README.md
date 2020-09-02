# Climate-Similarity-Metrics
Which similarity metrics are the most helpful to understand climate?

# Table of Contents

Notebooks:

0. Visualizes the data using the matplotlib basemap toolkit
1. Plots the mean per longitude of the u wind component for every latitude of every year
2. Computes the similarity to a random point for every point on the map
3. Derives different QBO indices from the data
4. Converts geographical coordinates into the corresponding indices for the gaussian grid
5. Computes the Mutual Information and Pearson Correlation between the QBO and the whole map at 30 hPa (for the whole period, for the whole period but every month seperately and for the whole period but only winter months are taken into account)
6. Lists the different similarity measures implemented in the similarity_measures.py file
7. Presents different functions to make similarity measures comparable
8. Presents different functions to combine the similarity measures
9. Visualizes the relationship between different similarity measures
10. Presents the notion of agreeableness between similarity measures 
10_a. Puts at disposal a widget to try different value- and agreement-threshold combinations
11. Presents the initial idea with level of agreement
12. Computes and plots the dependencies over time
13. Computes and plots the dependencies over different pressure levels 
14. Computes and plots the dependencies and the agreeableness between different similarity measures over time and levels
15. Shows some further applications 


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
