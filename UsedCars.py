#!/usr/bin/env python
# coding: utf-8

# ---
# 
# <center><h1> 📍 📍 Analyzing Selling Price of Used Cars 📍 📍 </h1></center>
# <center><h2> Arham Mirza </h2></center>
# 
# 
# ---
# 
# 

# ## Importing Libraries

# In[18]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp


# ## Checking the first five entries in the dataset

# In[20]:


df = pd.read_csv('output1.csv')

df = df.iloc[: , 1:]

df.head()


# ## Defining headers for the dataset

# In[44]:


headers = ["normalized-losses", "make", 
           "fuel-type", "aspiration","num-of-doors",
           "body-style","drive-wheels", "engine-location",
           "wheel-base","length", "width","height", "curb-weight",
           "engine-type","num-of-cylinders", "engine-size", 
           "fuel-system","bore","stroke", "compression-ratio",
           "horsepower", "peak-rpm","city-mpg","highway-mpg","price"]

df.columns=headers
df.head()


# ## Finding the missing value if any

# In[45]:


data = df

data.isna().any()

data.isnull().any() 


# ## Converting mpg to L/100km and checking the data type of each column

# In[46]:


data['city-mpg'] = 235 / df['city-mpg']
data.rename(columns = {'city_mpg': "city-L / 100km"}, inplace = True)

print(data.columns)

data.dtypes 


# ## Doing descriptive analysis of data categorical to numerical value

# In[47]:


pd.get_dummies(data['fuel-type']).head()

data.describe()


# ## Here, price is of object type(string), it should be int or float

# In[54]:


data.price.unique()

# Here it contains '?', so we Drop it
data = data[data.price != '?']

data['price'] = data['price'].astype(int)

# checking it again
data.dtypes


# ## Normalizing values by using simple feature scaling method and binning- grouping values
# 

# In[55]:


data['length'] = data['length']/data['length'].max()
data['width'] = data['width']/data['width'].max()
data['height'] = data['height']/data['height'].max()

# binning- grouping values
bins = np.linspace(min(data['price']), max(data['price']), 4) 
group_names = ['Low', 'Medium', 'High']
data['price-binned'] = pd.cut(data['price'], bins, 
                              labels = group_names, 
                              include_lowest = True)

print(data['price-binned'])
plt.hist(data['price-binned'])
plt.show()


# ## Grouping the data according to wheel, body-style and price.

# In[52]:


test = data[['drive-wheels', 'body-style', 'price']]
data_grp = test.groupby(['drive-wheels', 'body-style'], 
                         as_index = False).mean()

data_grp


# ## Plotting the data according to the price based on engine size

# In[56]:


plt.boxplot(data['price'])

sns.boxplot(x ='drive-wheels', y ='price', data = data)

plt.scatter(data['engine-size'], data['price'])
plt.title('Scatterplot of Enginesize vs Price')
plt.xlabel('Engine size')
plt.ylabel('Price')
plt.grid()
plt.show()

