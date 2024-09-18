#!/usr/bin/env python
# coding: utf-8

# In[4]:


import rasterio
from rasterio.plot import show

land_image_path = "la_eagleview_small.tif"
land_label_path = "training_data_ev.tif"

with rasterio.open(land_image_path) as src:
    land_image = src.read()
    show(land_image)
    height = src.height
    width = src.width
    bands = src.count  
    print("Image height: ",height)
    print("Image width: ",width)
    print("Number of bands:",bands)

with rasterio.open(land_label_path) as src:
    land_label = src.read()
    show(land_label)
    label_height = src.height
    label_width = src.width
    label_bands = src.count  
    print("Label image height: ",label_height)
    print("Label image width: ",label_width)
    print("Number of label image bands: ",label_bands)


# In[38]:


import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

with rasterio.open(land_image_path) as land_image, rasterio.open(land_label_path) as land_label:
    plt.figure(figsize = (15,15))
    # show the areial image
    blue = land_image.read(3, masked=True)
    green = land_image.read(2, masked=True)
    red = land_image.read(1, masked=True)
    rgb_land = np.dstack((red, green, blue))
    plt.imshow(rgb_land)
    # show the labeled training data above the image
    # create a color palette to color the labels. This is because we may want to use some intuitive 
    # colors to show the label (e.g., blue for pool) 
    palette = np.array([[0, 0, 0, 0], # no data
     [0, 255, 255,250], # pool
     [255, 255, 255,250], # street
     [100,238,100,250], #grass
     [255, 255, 0,250], #roof
     [0,100,0,250], #tree
     [0, 0, 255,250] #shadow
     ])
    land_label_data = land_label.read(1)
    plt.imshow(palette[land_label_data]) 

# Get the labeled training data for each band
red_train = red[land_label_data>0]
blue_train = blue[land_label_data>0]
green_train = green[land_label_data>0]
X_label = np.column_stack((red_train, blue_train,green_train)) # put the three features as 
# three columns of the 
#matrix
# Get the labeled value
y_label = land_label_data[land_label_data>0]
# Split to training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_label, y_label, test_size=0.2, 
random_state=42)


# In[39]:


from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(min_samples_leaf=10, random_state=42)
rf_model.fit(X_train, y_train)


# In[40]:


from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy of the model on the test data is: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix of the model:")
print(conf_matrix)


# In[43]:


# prepare all the pixels of this image
X_whole = np.column_stack((red.ravel(), blue.ravel(), green.ravel()))
y_whole_pred = rf_model.predict(X_whole)
# Reshape the prediction result into the shape of the image
y_whole_pred_reshape = y_whole_pred.reshape(land_image.height, land_image.width)
# show classification
plt.figure(figsize=(10,10))
plt.imshow(palette[y_whole_pred_reshape])


# In[ ]:




