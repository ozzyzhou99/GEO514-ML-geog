#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
world1 = gpd.read_file('SJER_plot_centroids.shp')
world1.plot()
world1.crs
# the coordinate reference system (crs) of “SJER_plot_centroids.shp”is WGS 84 / UTM zone 11N
# Unit: Meter
# Datum: World Geodetic System 1984 ensemble
# Prime meridian: Greenwich


# In[5]:


world2 = gpd.read_file('SJER_crop.shp')
print(world2)
world2.shape
# The shape of the SJER_crop.shp is (1,2) so there is only one row of data record in this file.


# In[2]:


import geopandas as gpd
import matplotlib.pyplot as plt
gpd.datasets.available
crop = gpd.read_file('SJER_crop.shp')
centroids = gpd.read_file('SJER_plot_centroids.shp')
fig, ax = plt.subplots(figsize=(8,8))
crop.plot(ax=ax, color="green", edgecolor="black")
centroids.plot(ax=ax, color="brown",markersize=10)


# In[4]:


import geopandas as gpd
centroids = gpd.read_file('SJER_plot_centroids.shp')
centroids['geometry'] = centroids.geometry.buffer(150)
#Use "centroids['geometry']"" can perform a buffer while maintaining the original attribute data
#Keep the buffer output as GeoDataframes instead of Geoseries
print(centroids)
centroids.plot()


# In[11]:


crop_diff = crop.overlay(centroids, how='difference')
crop_diff.plot()

