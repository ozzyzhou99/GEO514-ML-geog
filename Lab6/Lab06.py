#!/usr/bin/env python
# coding: utf-8

# In[2]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

nyc_neighborhoods = gpd.read_file('Data/Neighborhood.shp')
flickr_data = pd.read_csv('Data/Flickr_NewYork.csv', header=None)

# convert flickr
geometry = gpd.points_from_xy(flickr_data.iloc[:, 3], flickr_data.iloc[:, 2])
flickr_geo = gpd.GeoDataFrame(geometry=geometry)

plt.figure(figsize=(10,10))
ax = nyc_neighborhoods.plot(color='white', edgecolor='black')

flickr_geo.plot(ax=ax, color='red', markersize=1)

plt.title('NewYork Flickr Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[3]:


import os

# Set environment variables at the very top of your script
os.environ['OMP_NUM_THREADS'] = '2'  # Try reducing further if necessary
os.environ['MKL_NUM_THREADS'] = '2'  # Specific to MKL if it's being used
os.environ['NUMEXPR_NUM_THREADS'] = '2'  # For other libraries that use NumExpr

# Print to verify that the settings have been applied
print("OMP_NUM_THREADS:", os.getenv('OMP_NUM_THREADS'))
print("MKL_NUM_THREADS:", os.getenv('MKL_NUM_THREADS'))
print("NUMEXPR_NUM_THREADS:", os.getenv('NUMEXPR_NUM_THREADS'))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Extract latitude and longitude
X = flickr_data.iloc[:, [2, 3]]

k_values = range(5, 21)
silhouette_scores = []

# Iterate
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(k_values, silhouette_scores, marker='o', color='black', linestyle='--')
plt.title('Figure of K-means Silhoutte')
plt.xlabel('Number of k')
plt.ylabel('Silhouette')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[4]:


ax = nyc_neighborhoods.plot(color='white', edgecolor='black')

# K-MEANS clustering
kmeans_model = KMeans(n_clusters=13, random_state=42)
flickr_geo['cluster'] = kmeans_model.fit_predict(X)

#define color array
color_array = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
               '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324']

for cluster_id, color in zip(range(13), color_array):
    flickr_geo[flickr_geo['cluster'] == cluster_id].plot(ax=ax, color=color, markersize=2, alpha=0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[16]:


from sklearn.cluster import DBSCAN
import numpy as np

ax = nyc_neighborhoods.plot(color='white', edgecolor='black')

# DBSCAN clustering
eps = 0.002
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
flickr_geo['cluster'] = dbscan.fit_predict(X)

# Predict clusters
dbscan_label = dbscan.fit_predict(X)
cluster_count = len(set(dbscan_label)) - 1

# color array 
np.random.seed(42)
color_array_rand = []
for i in range(cluster_count):
    color_array_rand.append('#' + '%06X' % np.random.randint(0, 0xFFFFFF))

for cluster_id, color in zip(range(cluster_count), color_array_rand):
    flickr_geo[flickr_geo['cluster'] == cluster_id].plot(
        ax=ax, 
        color=color, 
        markersize=1, 
        alpha=0.5 
        )
    
# Plot noise points
noise_points = flickr_geo[flickr_geo['cluster'] == -1]
noise_points.plot(ax=ax, color='black', markersize=2, alpha=0.5)

plt.title('DBSCAN Clustering Result')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[7]:


from sklearn.cluster import DBSCAN
import numpy as np

ax = nyc_neighborhoods.plot(color='white', edgecolor='black')

# DBSCAN clustering
eps = 0.002
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
flickr_geo['cluster'] = dbscan.fit_predict(X)

# Predict clusters
dbscan_label = dbscan.fit_predict(X)
cluster_count = len(set(dbscan_label)) - 1

# color array 
np.random.seed(42)
color_array_rand = []
for i in range(cluster_count):
    color_array_rand.append('#' + '%06X' % np.random.randint(0, 0xFFFFFF))

for cluster_id, color in zip(range(cluster_count), color_array_rand):
    flickr_geo[flickr_geo['cluster'] == cluster_id].plot(ax=ax, color=color, markersize=2, alpha=0.5)

# Plot noise points
noise_points = flickr_geo[flickr_geo['cluster'] == -1]
noise_points.plot(ax=ax, color='black', markersize=5, alpha=0.5)

plt.axis([-74.05, -73.92, 40.67, 40.83])

plt.title('DBSCAN Clustering Result')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[ ]:




