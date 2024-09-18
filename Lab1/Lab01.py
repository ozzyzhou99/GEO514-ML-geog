#!/usr/bin/env python
# coding: utf-8

# In[24]:


import folium
from geopy import distance

#Code for task 1
cities = {}
f = open('cities.txt')
line = f.readline()
clats = []
clons = []
while line:
    line = line.rstrip('\n')
    cities_list = line.split('|')
    cname = cities_list[0]
    clat = float(cities_list[1])
    clon = float(cities_list[2])
    cities[cname] = [clat, clon]
    clats.append(clat)
    clons.append(clon)
    line = f.readline()
f.close()
#print(cities)
    
#Code for task 2
min_lat=min(clats)
max_lat=max(clats)
min_lon=min(clons)
max_lon=max(clons)
a = (min_lon+max_lon)/2
b = (min_lat+max_lat)/2
m = folium.Map(
    max_bounds=True,
    location=[a, b],
    zoom_start=5,
    min_lat=min(clats)-2,
    max_lat=max(clats)+2,
    min_lon=min(clons)-2,
    max_lon=max(clons)+2,
)

group_1 = folium.FeatureGroup("first group").add_to(m)
for value in cities.values():
    folium.Marker(value, icon=folium.Icon("red")).add_to(group_1)


#Code for task 3
citiesD = cities
outputfile = open("city_distance.txt","w")
for key,value in list(cities.items()):
    cnameA = key
    locationA = (value[0],value[1])
    del(citiesD[cnameA])
    for key,value in list(citiesD.items()):
        cnameB = key
        locationB = (value[0],value[1])
        if cnameA != cnameB:
            d = distance.distance(locationA, locationB).km
            outputfile.write( str(cnameA) + "|" + str(cnameB) + "|" + str(d) + " km\n")
outputfile.close()

m


# In[ ]:




