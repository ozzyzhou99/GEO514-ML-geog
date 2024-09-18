#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import folium

gdf = gpd.read_file("california_housing.shp")

print(gdf.head(10))

m = folium.Map(location=[37.8, -122.0], zoom_start=6)  


for index, row in gdf.sample(n=500, random_state=42).iterrows():
    folium.Marker(location=[row.geometry.y, row.geometry.x]).add_to(m)

m


# In[4]:


import matplotlib.pyplot as plt
import geopandas as gpd

gdf = gpd.read_file("california_housing.shp")
plt.hist(gdf["house_age"],bins=50)
plt.xlabel('House Age')
plt.ylabel('Frequency')
plt.title('Histogram of House Age')
plt.show()                                 # show original  House age histogram
hamax = gdf["house_age"].max()
gdf_cleaned = gdf[gdf["house_age"] < hamax]
plt.hist(gdf_cleaned["house_age"],bins=50)
plt.xlabel('House Age')
plt.ylabel('Frequency')
plt.title('Histogram of House Age')
plt.show()                                 # show cleaned histogram

gdf = gpd.read_file("california_housing.shp")
plt.hist(gdf["median_val"],bins=50)
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Histogram of Median House Value')
plt.show()                                 # show original histogram
mvmax = gdf["median_val"].max()
gdf_cleaned = gdf[gdf["median_val"] < mvmax]
plt.hist(gdf_cleaned["median_val"],bins=50)
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Histogram of Median House Value')
plt.show()                                 # show cleaned histogram


# In[5]:


gdf_cleaned["ocean_prox"].unique()

import pandas as pd
gdf_cleaned = pd.get_dummies(gdf_cleaned)


# In[4]:


gdf_cleaned['x'] = gdf_cleaned['geometry'].x
gdf_cleaned['y'] = gdf_cleaned['geometry'].y   # Extract coordinate values
# Remove the attribute geometry
gdf_cleaned = gdf_cleaned[['total_room','total_bedr','population','households','median_inc','house_age','median_val','ocean_prox_<1H OCEAN','ocean_prox_INLAND','ocean_prox_ISLAND','ocean_prox_NEAR BAY','ocean_prox_NEAR OCEAN','x','y']]
gdf_cleaned.head()


# In[5]:


gdf_cleaned['ocean_prox_<1H OCEAN'] = gdf_cleaned['ocean_prox_<1H OCEAN'].astype(int)
gdf_cleaned['ocean_prox_INLAND'] = gdf_cleaned['ocean_prox_INLAND'].astype(int)
gdf_cleaned['ocean_prox_ISLAND'] = gdf_cleaned['ocean_prox_ISLAND'].astype(int)
gdf_cleaned['ocean_prox_NEAR BAY'] = gdf_cleaned['ocean_prox_NEAR BAY'].astype(int)
gdf_cleaned['ocean_prox_NEAR OCEAN'] = gdf_cleaned['ocean_prox_NEAR OCEAN'].astype(int)
# Convert Boolean values to integers so that describe in subsequent steps will not directly remove these Boolean values.


# In[6]:


gdf_cleaned.head()


# In[14]:


gdf_cleaned['ocean_prox_<1H OCEAN'] = gdf_cleaned['ocean_prox_<1H OCEAN'].astype(float)
gdf_cleaned['ocean_prox_INLAND'] = gdf_cleaned['ocean_prox_INLAND'].astype(float)
gdf_cleaned['ocean_prox_ISLAND'] = gdf_cleaned['ocean_prox_ISLAND'].astype(float)
gdf_cleaned['ocean_prox_NEAR BAY'] = gdf_cleaned['ocean_prox_NEAR BAY'].astype(float)
gdf_cleaned['ocean_prox_NEAR OCEAN'] = gdf_cleaned['ocean_prox_NEAR OCEAN'].astype(float)
#Convert to float to ensure that all data types are consistent
gdf_cleaned.head()


# In[15]:


training_data = gdf_cleaned.sample(frac=0.8, random_state=42)
test_data = gdf_cleaned.drop(training_data.index)
training_data.head()
# Split training data and test data


# In[19]:


training_X = training_data[['total_room','total_bedr','population','households','median_inc','house_age','ocean_prox_<1H OCEAN','ocean_prox_INLAND','ocean_prox_ISLAND','ocean_prox_NEAR BAY','ocean_prox_NEAR OCEAN','x','y']]
training_y = training_data['median_val']
test_X = test_data[['total_room','total_bedr','population','households','median_inc','house_age','ocean_prox_<1H OCEAN','ocean_prox_INLAND','ocean_prox_ISLAND','ocean_prox_NEAR BAY','ocean_prox_NEAR OCEAN','x','y']]
test_y = test_data['median_val']


# In[20]:


def standardize_data(data, stats):
    return (data - stats['mean'])/stats['std']


# In[22]:


training_stats = training_X.describe().transpose()
training_X_std = standardize_data(training_X, training_stats)
test_X_std = standardize_data(test_X, training_stats)
training_X.describe()


# In[23]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(training_X_std, training_y)


# In[24]:


test_y_pred = linear_model.predict(test_X_std)


# In[31]:


import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(test_y,test_y_pred,s=15, c='blue')
plt.xlabel("Observed values")
plt.ylabel("Predicted values")
plt.plot()
plt.plot([test_y.min(),test_y.max()],[test_y.min(),test_y.max()],'r--')


# In[34]:


from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(test_y,test_y_pred,squared=False)
rmse


# In[ ]:




