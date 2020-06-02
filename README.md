# Climate-Similarity-Metrics
Which similarity metrics are the most helpful to understand climate?

## Goal of the Project
Create a modular framework for analysing and understanding relationships in climate data. The moduls should allow for testing different climate indices, similarity metrics, and time-scales. The results should be interpretable by climate scientists.

## Data Description

The the *u* component (east-west) of the wind for 512 longitudes, 256 latitudes, and 37 altitudes aggregated by months.

## Data Setup

1. Download `era-int_pl_1981-mm-u.nc` file from this [repository](https://nextcloud.scc.kit.edu/s/cwpp3wdQPcm96jq).
2. Place the data file `era-int_pl_1981-mm-u.nc` in the directory `data/`.
3. Use `era-int_pl_1979-2019-mm-u.nc` file from this [repository](https://nextcloud.scc.kit.edu/s/cwpp3wdQPcm96jq) for the "Data Preprocessing" step

##  Data Preprocessing
Use the data manipulation tool cdo to extract all values for pressure level 30 hPa with the following command:

`cdo -select,level=30 era-int_pl_1979-2019-mm-u.nc era-int_pl_1979-2019-mm-l30-u.nc`

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
