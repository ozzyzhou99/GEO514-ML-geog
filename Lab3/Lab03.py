#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rasterio
data_DSM = rasterio.open("pre_DSM.tif")
data_DTM = rasterio.open("pre_DTM.tif")

print("The band number of DTM is: ",data_DTM.count)
print("The band number of DSM is: ",data_DSM.count)


# In[2]:


print("The height and width of DTM are: ", data_DTM.height, ",", data_DTM.width)
print("The height and width of DSM are: ", data_DSM.height, ",", data_DSM.width)


# In[3]:


from rasterio.plot import show
show(data_DTM)
show(data_DSM)


# In[8]:


import rasterio
data_DSM = rasterio.open("pre_DSM.tif")
data_DTM = rasterio.open("pre_DTM.tif")
DTM_read = data_DTM.read(1,masked = True)
DSM_read = data_DSM.read(1,masked = True)
data_output = DSM_read - DTM_read
print("The minimum height:",data_output.min())
print("The maximum height:",data_output.max())


# In[10]:


import matplotlib.pyplot as plt
plt.hist(data_output.ravel(),bins=range(0,31,2), color="black", edgecolor="white")
plt.xlabel("Height")
plt.ylabel("Frequency")


# In[13]:


import numpy as np
bins = np.array([0,2,7,12])
trees = np.digitize(data_output, bins)
show(trees)


# In[16]:


dataset_output = rasterio.open("trees.tif","w",driver = "GTiff",
                       width = 4000,
                       height = 2000,
                       count = 1,
                       dtype = data_DTM.read(1).dtype,
                       crs = data_DTM.crs,
                       transform = data_DTM.transform
                      )
trees_output = trees.astype(data_DTM.read(1).dtype)
dataset_output.write(trees_output,1)
dataset_output.close()


# In[ ]:




