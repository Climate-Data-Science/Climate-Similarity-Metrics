# Climate-Similarity-Metrics
Which similarity metrics are the most helpful to understand climate?

## Goal of the Project
Create a modular framework for analysing and understanding relationships in climate data. The moduls should allow for testing different climate indices, similarity metrics, and time-scales. The results should be interpretable by climate scientists.

## Data Description

The the *u* component (east-west) of the wind for 512 longitudes, 256 latitudes, and 37 altitudes aggregated by months.

## Data Setup

1. Download `era-int_pl_1981-mm-u.nc` file from this [repository](https://nextcloud.scc.kit.edu/s/cwpp3wdQPcm96jq).
2. Place the data file `era-int_pl_1981-mm-u.nc` in the directory `data/`.

##  Data Preprocessing
Use the data manipulation tool cdo to extract all values for pressure level 70 hPa with the following command:

`cdo -select,level=70 era-int_pl_1979-2019-mm-u.nc era-int_pl_1979-2019-mm-l70-u.nc`

## QBO Index

Values at Singapore (1N, 104E) at pressure level 70 hPa

### Data from the Internet to cross check

https://www.gfd-dennou.org/arch/eriko/QBO/index.html

## Useful Materials
* [How to Write a Good Git Commit Message](https://chris.beams.io/posts/git-commit/)
* [Python Styleguide by Google](http://google.github.io/styleguide/pyguide.html)
