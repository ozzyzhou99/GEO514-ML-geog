# GEO514 Machine Learning Geog Course Labs

This repository contains Python labs completed as part of the GEO514 course, focusing on geospatial data analysis using various techniques and tools in Python. The labs cover topics from vector and raster data manipulation to machine learning and spatial clustering.

## Table of Contents
1. [Lab 1: Geodesic Distance Calculation](#lab-1-geodesic-distance-calculation)
2. [Lab 2: Vector Data Analysis](#lab-2-vector-data-analysis)
3. [Lab 3: Raster Data Analysis and Reclassification](#lab-3-raster-data-analysis-and-reclassification)
4. [Lab 4: Predicting Median Housing Prices using Regression](#lab-4-predicting-median-housing-prices-using-regression)
5. [Lab 5: Land Cover Classification using Random Forest](#lab-5-land-cover-classification-using-random-forest)
6. [Lab 6: Geospatial Clustering of Geotagged Data](#lab-6-geospatial-clustering-of-geotagged-data)

## Lab 1: Geodesic Distance Calculation

In this lab, we calculate the geodesic distance between 106 US cities based on their latitude and longitude. The tasks include reading city coordinates, visualizing the cities on a map, and calculating the distances between all city pairs using the `geopy` and `folium` libraries.

- **Libraries used**: `geopy`, `folium`
- **Input data**: `cities.txt` containing city names and coordinates.
- **Output**: `city_distance.txt` with the distances between all city pairs.

## Lab 2: Vector Data Analysis

This lab involves analyzing vector data using two shapefiles: a point shapefile representing field sites and a polygon shapefile for the study area. Tasks include coordinate reference system (CRS) analysis, record counting, and geometric operations such as buffering and overlay differences.

- **Libraries used**: `GeoPandas`
- **Input data**: `SJER_plot_centroids.shp`, `SJER_crop.shp`
- **Tasks**: Buffer creation, overlay operations, and CRS identification.

## Lab 3: Raster Data Analysis and Reclassification

In this lab, we work with raster datasets, specifically Digital Terrain Models (DTM) and Digital Surface Models (DSM), to calculate tree heights and classify them based on height. We use `numpy` to reclassify raster data and visualize the results.

- **Libraries used**: `rasterio`, `numpy`
- **Input data**: `pre_DTM.tif`, `pre_DSM.tif`
- **Tasks**: Raster reclassification, histogram creation, and height analysis.

## Lab 4: Predicting Median Housing Prices using Regression

This lab focuses on predicting the median housing prices in California using a linear regression model. The tasks involve cleaning and preprocessing geospatial data, converting categorical variables to dummy variables, and evaluating model performance using Root Mean Squared Error (RMSE).

- **Libraries used**: `GeoPandas`, `pandas`, `scikit-learn`
- **Input data**: `California_housing.shp`
- **Tasks**: Data preprocessing, regression model training, and performance evaluation.

## Lab 5: Land Cover Classification using Random Forest

In this lab, we use a random forest model to classify land cover types from an aerial image. The model is trained on human-annotated data, and then applied to classify the entire image into categories such as grass, streets, trees, etc.

- **Libraries used**: `rasterio`, `scikit-learn`
- **Input data**: `la_eagleview_small.tif`, `training_data_ev.tif`
- **Tasks**: Data preparation, random forest model training, and classification visualization.

## Lab 6: Geospatial Clustering of Geotagged Data

This lab uses geotagged Flickr photo locations in New York City to perform spatial clustering. We explore clustering methods like K-Means and DBSCAN to identify areas of interest based on photo density.

- **Libraries used**: `GeoPandas`, `scikit-learn`, `pandas`
- **Input data**: `Neighborhood.shp`, `Flickr_NewYork.csv`
- **Tasks**: Spatial clustering using K-Means and DBSCAN, silhouette analysis.

## Installation

To run these labs, ensure you have the following Python packages installed:

```bash
pip install geopandas rasterio folium geopy scikit-learn numpy matplotlib pandas
```

##Usage

Clone the repository and run the .py or .ipynb files using a Python environment or Jupyter Notebook. Each lab includes detailed code and comments explaining the tasks. Ensure the input data files are available in the expected directories.
```bash
git clone <repository-url>
cd <repository-folder>
```
